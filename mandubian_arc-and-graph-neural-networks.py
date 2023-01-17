!pip install torch_geometric
!pip install torch_sparse
!pip install torch_scatter
!pip install pytorch_lightning
!pip install wandb
import numpy as np
import pandas as pd

import os
import json
from pathlib import Path
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F

from torch_geometric.utils import to_networkx

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from typing import Tuple, Dict, List

import networkx as nx

%matplotlib inline
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
print("training_path", training_path)
print("evaluation_path", evaluation_path)
print("test_path", test_path)
# COPIED from https://github.com/rusty1s/pytorch_geometric
# and very lightly modified to use `non_blocking` param for Tensor.to(...)
# Maybe I'll do a PR with it
import re
import copy
import warnings

import torch
import torch_geometric
from torch_sparse import coalesce, SparseTensor
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

from torch_geometric.utils.num_nodes import maybe_num_nodes

__num_nodes_warn_msg__ = (
    'The number of nodes in your data object can only be inferred by its {} '
    'indices, and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated nodes. Please consider explicitly setting '
    'the number of nodes for this data object by assigning it to '
    'data.num_nodes.')


def size_repr(key, item, indent=0):
    indent_str = ' ' * indent
    if torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, SparseTensor):
        out = str(item.sizes())[:-1] + f', nnz={item.nnz()}]'
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + indent_str + '}'
    else:
        out = str(item)

    return f'{indent_str}{key}={out}'


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)
    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    Example::
        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.norm = norm
        self.face = face
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != torch.long:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

        if torch_geometric.is_debug_enabled():
            self.debug()

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        if torch_geometric.is_debug_enabled():
            data.debug()

        return data

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if bool(re.search('(index|face)', key)) else 0

    def __inc__(self, key, value):
        r""""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        if self.face is not None:
            warnings.warn(__num_nodes_warn_msg__.format('face'))
            return maybe_num_nodes(self.face)
        if self.edge_index is not None:
            warnings.warn(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    def is_coalesced(self):
        r"""Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        edge_index, _ = coalesce(self.edge_index, None, self.num_nodes,
                                 self.num_nodes)
        return self.edge_index.numel() == edge_index.numel() and (
            self.edge_index != edge_index).sum().item() == 0

    def coalesce(self):
        r""""Orders and removes duplicated entries from edge indices."""
        self.edge_index, self.edge_attr = coalesce(self.edge_index,
                                                   self.edge_attr,
                                                   self.num_nodes,
                                                   self.num_nodes)
        return self

    def contains_isolated_nodes(self):
        r"""Returns :obj:`True`, if the graph contains isolated nodes."""
        return contains_isolated_nodes(self.edge_index, self.num_nodes)

    def contains_self_loops(self):
        """Returns :obj:`True`, if the graph contains self-loops."""
        return contains_self_loops(self.edge_index)

    def is_undirected(self):
        r"""Returns :obj:`True`, if graph edges are undirected."""
        return is_undirected(self.edge_index, self.edge_attr, self.num_nodes)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def __apply__(self, item, func):
        if torch.is_tensor(item) or isinstance(item, SparseTensor):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    #def to(self, device, *keys, **kwargs):
    #    r"""Performs tensor dtype and/or device conversion to all attributes
    #    :obj:`*keys`.
    #    If :obj:`*keys` is not given, the conversion is applied to all present
    #    attributes."""
    #    return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def to(self, device, non_blocking, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device=device, non_blocking=non_blocking, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })

    def debug(self):
        if self.edge_index is not None:
            if self.edge_index.dtype != torch.long:
                raise RuntimeError(
                    ('Expected edge indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.edge_index.dtype))

        if self.face is not None:
            if self.face.dtype != torch.long:
                raise RuntimeError(
                    ('Expected face indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.face.dtype))

        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    ('Edge indices should have shape [2, num_edges] but found'
                     ' shape {}').format(self.edge_index.size()))

        if self.edge_index is not None and self.num_nodes is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    ('Face indices should have shape [3, num_faces] but found'
                     ' shape {}').format(self.face.size()))

        if self.face is not None and self.num_nodes is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Face indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    ('Edge indices and edge attributes hold a differing '
                     'number of edges, found {} and {}').format(
                         self.edge_index.size(), self.edge_attr.size()))

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node features should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.x.size(0)))

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node positions should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.pos.size(0)))

        if self.norm is not None and self.num_nodes is not None:
            if self.norm.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node normals should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.norm.size(0)))

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))
# COPIED from https://github.com/rusty1s/pytorch_geometric
# Maybe I'll do a PR with it
import torch
import torch_geometric
#from torch_geometric.data import Data


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
# To refactor all of this code because my aim was just to be quick there
# Remark that I code Python with types but here types are not everywhere by lack of time ;)

def sample_to_graph(sample):
    """
    ARC sample to Geometric Graph.
    
    Every pixel becomes a node and it has edges to its neighbor pixels.
    """
    height = len(sample)
    width = len(sample[0])
    data_x = torch.tensor(sample, dtype=torch.float)
    data_x = data_x.flatten().unsqueeze(dim=1).contiguous()
    edge_index=[]
    pos = []
    # terrible code now...
    for y in range(height-1, -1, -1):
        for x in range(width):
            local_edges = []
            pos.append([x, y])
            x0 = x - 1
            y0 = y - 1
            x1 = x + 1
            y1 = y + 1
            pt = x + width * y
            if x0 >= 0:
                local_edges.append([x0 + width * y, pt])
                if y0 >= 0:
                    local_edges.append([x0 + width * y0, pt])
                if y1 <= height - 1:
                    local_edges.append([x0 + width * y1, pt])
                                                       
            if x1 <= width - 1:
                local_edges.append([x1 + width * y, pt])
                if y0 >= 0:
                    local_edges.append([x1 + width * y0, pt])
                if y1 <= height - 1:
                    local_edges.append([x1 + width * y1, pt])

            if y0 >= 0:
                local_edges.append([x + width * y0, pt])
            if y1 <= height - 1:
                local_edges.append([x + width * y1, pt])
            
            edge_index.extend(local_edges)
                    
    edge_index = torch.tensor(edge_index, dtype=torch.long) #
    pos = torch.tensor(pos, dtype=torch.long)
    data = Data(x=data_x, edge_index=edge_index.t().contiguous(), pos=pos)
    return data

def sample_to_graph_padded(sample, target, pad_value = 10) -> Data:
    """
    ARC sample to Geometric Graph padding sample to fit size of target if it is bigger.
    
    Fake nodes are linked to other nodes exactly as pixels and their value is
    pad_value (which is 10 to be out of ARC classes).
    """
    target_height = len(target)
    target_width = len(target[0])
    
    height = len(sample)
    width = len(sample[0])
    
    # same size
    if target_height == height and target_width == width:
        return sample_to_graph(sample)
    # rest is padding
    else:
        data_x = torch.tensor(sample, dtype=torch.float)

        diff_height = max(target_height, height) - height
        if diff_height % 2 == 0:
            pad_top = pad_bottom = int(diff_height / 2)
        else:
            pad_top = int(diff_height / 2)
            pad_bottom = pad_top + 1
            
        diff_width = max(target_width, width) - width
        if diff_width % 2 == 0:
            pad_left = pad_right = int(diff_width / 2)
        else:
            pad_left =int(diff_width / 2)
            pad_right = pad_left + 1

        
        data_x = F.pad(data_x, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
        data_x = data_x.flatten().unsqueeze(dim=1).contiguous()
        edge_index=[]
        pos = []
        final_height = max(target_height, height)
        final_width = max(target_width, width)
        for y in range(final_height-1, -1, -1):
            for x in range(final_width):
                local_edges = []
                pos.append([x, y])
                x0 = x - 1
                y0 = y - 1
                x1 = x + 1
                y1 = y + 1
                pt = x + final_width * y
                if x0 >= 0:
                    local_edges.append([x0 + final_width * y, pt])
                    if y0 >= 0:
                        local_edges.append([x0 + final_width * y0, pt])
                    if y1 <= final_height - 1:
                        local_edges.append([x0 + final_width * y1, pt])

                if x1 <= final_width - 1:
                    local_edges.append([x1 + final_width * y, pt])
                    if y0 >= 0:
                        local_edges.append([x1 + final_width * y0, pt])
                    if y1 <= final_height - 1:
                        local_edges.append([x1 + final_width * y1, pt])

                if y0 >= 0:
                    local_edges.append([x + final_width * y0, pt])
                if y1 <= final_height - 1:
                    local_edges.append([x + final_width * y1, pt])

                edge_index.extend(local_edges)

        edge_index = torch.tensor(edge_index, dtype=torch.long) #
        pos = torch.tensor(pos, dtype=torch.long)
        data = Data(x=data_x, edge_index=edge_index.t().contiguous(), pos=pos)
        return data

def build_image_edges(final_height: int, final_width: int,
                      start_height: int, start_width: int,
                      end_height: int, end_width: int):
    """
    Utilities to build all edges for an image to graph in which pixel are nodes
    and pixel is linked to all neighbors.
    """
    edge_index=[]
    #final_height = start_height + final_height
    #final_width = start_width + final_width
    for y in range(end_height-1, start_height-1, -1):
        for x in range(start_width, end_width):
            local_edges = []
            x0 = x - 1
            y0 = y - 1
            x1 = x + 1
            y1 = y + 1
            pt = x + final_width * y
            if x0 >= start_width:
                local_edges.append([x0 + final_width * y, pt])
                if y0 >= start_height:
                    local_edges.append([x0 + final_width * y0, pt])
                if y1 <= end_height - 1:
                    local_edges.append([x0 + final_width * y1, pt])

            if x1 <= end_width - 1:
                local_edges.append([x1 + final_width * y, pt])
                if y0 >= start_height:
                    local_edges.append([x1 + final_width * y0, pt])
                if y1 <= end_height - 1:
                    local_edges.append([x1 + final_width * y1, pt])

            if y0 >= start_height:
                local_edges.append([x + final_width * y0, pt])
            if y1 <= end_height - 1:
                local_edges.append([x + final_width * y1, pt])

            edge_index.extend(local_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long) #
    
    return edge_index
    
def build_image_pos(final_height: int, final_width: int):
    """Building basic image positions for graph drawing."""
    pos = []
    for y in range(final_height-1, -1, -1):
        for x in range(final_width):
            pos.append([x, y])
            
    pos = torch.tensor(pos, dtype=torch.long)
    
    return pos
    
def build_edges_image_to_image(final_height: int, final_width: int):
    """
    In the context of input/output graphs merged horizontally, connect both graphs
    by drawing edges from every node of graph to every node of other graph.
    """
    edge_index=[]
    for y in range(final_height-1, -1, -1):
        for x in range(final_width):
            edge_index.append([x + 2*final_width * y, x + 2*final_width * y + final_width])
            edge_index.append([x + 2*final_width * y + final_width, x + 2*final_width * y])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index
    
def sample_to_graph_merged(sample, target, empty_target: bool=False, pad_value = 10) -> Data:
    """Concat sample & target padded graphs horizontally and draw edges between every node."""
    target_height = len(target)
    target_width = len(target[0])
    
    height = len(sample)
    width = len(sample[0])
    
    # Pad Sample
    data_x = torch.tensor(sample, dtype=torch.float)
    diff_height = max(target_height, height) - height
    if diff_height % 2 == 0:
        pad_top = pad_bottom = int(diff_height / 2)
    else:
        pad_top = int(diff_height / 2)
        pad_bottom = pad_top + 1

    diff_width = max(target_width, width) - width
    if diff_width % 2 == 0:
        pad_left = pad_right = int(diff_width / 2)
    else:
        pad_left =int(diff_width / 2)
        pad_right = pad_left + 1

    data_x = F.pad(data_x, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
    
    # Pad Target
    data_y = torch.tensor(target, dtype=torch.float)
    diff_target_height = max(target_height, height) - target_height
    if diff_target_height % 2 == 0:
        pad_target_top = pad_target_bottom = int(diff_target_height / 2)
    else:
        pad_target_top = int(diff_target_height / 2)
        pad_target_bottom = pad_target_top + 1

    diff_target_width = max(target_width, width) - target_width
    if diff_target_width % 2 == 0:
        pad_target_left = pad_target_right = int(diff_target_width / 2)
    else:
        pad_target_left =int(diff_target_width / 2)
        pad_target_right = pad_target_left + 1
    data_y = F.pad(data_y, pad=(pad_target_left, pad_target_right, pad_target_top, pad_target_bottom), mode='constant', value=pad_value)
    if empty_target:
        data_y = data_y.fill_(pad_value)

    # horizontally cat
    data_full = torch.cat((data_x, data_y), dim=1)
    data_full = data_full.flatten().unsqueeze(dim=1).contiguous()

    final_height = max(target_height, height)
    final_width = max(target_width, width) # twice as we concat horizontally

    edges_index_x = build_image_edges(final_height, 2*final_width, 0, 0, final_height, final_width)
    edges_index_y = build_image_edges(final_height, 2*final_width, 0, final_width, final_height, 2*final_width)
    pos = build_image_pos(final_height, 2*final_width)
    edges_image_to_image = build_edges_image_to_image(final_height, final_width)
    #edges_index = torch.cat((edges_index_x, edges_index_y, edges_image_to_image), dim=0)
    edges_index = torch.cat((edges_index_x, edges_index_y, edges_image_to_image), dim=0)
    #pos = torch.cat((pos_x, pos_y), dim=0)

    data = Data(x=data_full, edge_index=edges_index.t().contiguous(), pos=pos)
    return data
    
    


# Code copied from master version because the one in v1.4.3 is buggy
def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.
    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]
    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G

DEFAULT_CMAP = [
    '#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
    #_1 color
    '#FFFFFF'
]

def draw_graph(data: Data, title: str, ax=None, cmap=DEFAULT_CMAP):
    """Draw simple graph"""
    G = to_networkx(data, to_undirected=True, node_attrs=["x", "pos"])
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    #pos = nx.kamada_kawai_layout(nx_G)
    node_labels = nx.get_node_attributes(G, 'x')
    node_pos = nx.get_node_attributes(G, 'pos')
    pos = nx.spring_layout(G, pos=node_pos) #, fixed = fixed_nodes)
    node_labels = { i: int(x) for (i, x) in node_labels.items()}
    nx.draw(G, pos, ax=ax)
    node_pos = { i:i for i, _ in node_pos.items() }
    nx.draw_networkx_nodes(G, pos, node_color=[cmap[i] for i in node_labels.values()], ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_pos, font_color="w", ax=ax)
    if ax is not None:
        ax.set_title(title)
    else:
        plt.title(title)


    
def draw_batch(batch: Batch, title: str, figsize=(15, 5), cmap=DEFAULT_CMAP):
    """Draw batch of graphs"""
    data_list = batch.to_data_list()
    fig, axs = plt.subplots(len(data_list), 2, figsize=figsize)
    plt.box(False)
    for idx, data in enumerate(data_list):
        draw_graph(data, f"{title}_{idx}_input", axs[idx, 0], cmap)
        draw_graph(data.y[0], f"{title}_{idx}_output", axs[idx, 1], cmap)

    #plt.savefig('this.png')
    plt.show()

    
def draw_batch_images(batch_images, title: str, use_pad: bool = True, merging: bool = False, figsize=(15, 5), cmap=DEFAULT_CMAP):
    """Draw batch from ARC dataset."""
    norm = colors.Normalize(vmin=0, vmax=9)

    if not merging:
        batch, batch_out, batch_pad, batch_out_pad, images = batch_images

        if use_pad:
            data_list = batch_pad.to_data_list()
            data_out_list = batch_out_pad.to_data_list()
        else:
            data_list = batch.to_data_list()
            data_out_list = batch_out.to_data_list()
    else:
        batch, batch_out, batch_pad, batch_out_pad, batch_merged, batch_merged_out, images = batch_images
        data_list = batch_pad.to_data_list()
        data_out_list = batch_out_pad.to_data_list()
        
    lg = len(data_list)+ len(images)
    figsize = (figsize[0], figsize[1] * lg)
    fig, axs = plt.subplots(lg, 2, figsize=figsize)
    plt.box(False)
    for idx, (data, data_out, imgs) in enumerate(zip(data_list, data_out_list, images)):
        axs[2*idx, 0].imshow(imgs['input'], cmap=colors.ListedColormap(cmap), norm=norm)
        axs[2*idx, 0].set_title(f"{title}_{idx}_input")
        #axs[idx, 0].set_title('Train Input')
        draw_graph(data, f"{title}_{idx}_input", axs[2*idx+1, 0], cmap)
        axs[2*idx, 1].imshow(imgs['output'], cmap=colors.ListedColormap(cmap), norm=norm)
        axs[2*idx, 1].set_title(f"{title}_{idx}_output")
        draw_graph(data_out, f"{title}_{idx}_output", axs[2*idx+1, 1], cmap)

    #draw_graph(batch_pad, f"{title}_batch_input", axs[lg-1, 1], cmap)
    #draw_graph(batch_out_pad, f"{title}_batch_output", axs[lg, 1], cmap)

    #plt.savefig('this.png')
    plt.tight_layout()
    plt.show()
task_file = str(training_path / '007bbfb7.json')

with open(task_file, 'r') as f:
    task = json.load(f)

print(task.keys())
data_in = sample_to_graph(task['train'][0]["input"])
data_out = sample_to_graph(task['train'][0]["output"])
print("data_in", data_in)
print("data_out", data_out)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
plt.box(False)
draw_graph(data_in, "input", axs[0])
draw_graph(data_out, "output", axs[1])

#plt.savefig('this.png')
plt.show()
data_in_padded = sample_to_graph_padded(task['train'][0]["input"], task['train'][0]["output"])
data_out_padded = sample_to_graph(task['train'][0]["output"])

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
plt.box(False)
draw_graph(data_in_padded, "input", axs[0])
draw_graph(data_out_padded, "output", axs[1])

#plt.savefig('this.png')
plt.show()
data_merged = sample_to_graph_merged(task['train'][0]["input"], task['train'][0]["output"])

#fig, axs = plt.subplots(1, 2, figsize=(15, 5))
plt.figure(figsize=(20,10)) 
plt.box(False)
draw_graph(data_merged, "merged")
plt.show()
import os.path as osp
import torch
from torch_geometric.data import Dataset
import torch_geometric
from torch._six import container_abcs, string_classes, int_classes


class ARCGeometricDataset(Dataset):
    """
    ARC Geometric Dataset loads all ARC tasks in memory and builds a Geometric Batch aggregated graph
    for each task.
    Each element of the dataset is tuple of a Geometric Batch of all train samples in task and original images.
    
    Default class for padding of graph is 11 (10 + 1)
    """
    def __init__(self, arc_path: str, root, mode: str = "train", transform=None, pre_transform=None,
                 pad_value=10, merging: bool = False):
        self.arc_path = Path(arc_path)
        self.mode = mode
        self.task_files = [self.arc_path / f for f in sorted(os.listdir(arc_path))]
        self.tasks = [task_file.stem for task_file in self.task_files]
        self.graph_files = [Path(root) / f"{task}.pt" for task in self.tasks]
        self.pad_value = pad_value
        self.merging = merging
        super(ARCGeometricDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.tasks

    @property
    def processed_file_names(self):
        return self.graph_files

    @property
    def raw_dir(self):
        return  self.arc_path    
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0][0].num_node_features    
    #def download(self):
        # Download to `self.raw_dir`.

        
    def process(self):
        batches = []
        batches_out = []
        batches_pad = []
        batches_out_pad = []
        originals = []
        batches_merged = []
        batches_merged_out = []
        nb = 0
        for (task, task_file, graph_file) in zip(self.tasks, self.task_files, self.graph_files):
            with open(task_file, 'r') as f:
                data_batch = []
                data_batch_out = []
                data_batch_pad = []
                data_batch_out_pad = []
                img_batch = []
                task = json.load(f)
                for t in task["train"]:
                    img_batch.append(t)
                    data = sample_to_graph(t["input"])
                    data_out = sample_to_graph(t["output"])
                    data_batch_out.append(data_out)
                    
                    # pad input to output
                    data_pad = sample_to_graph_padded(t["input"], t["output"], pad_value = self.pad_value)
                    # pad output to input to be sure to have same size at the end
                    data_out_pad = sample_to_graph_padded(t["output"], t["input"], pad_value = self.pad_value)
                    #data_pad.y = data_out_pad
                    data_batch_out_pad.append(data_out_pad)

                    data_batch.append(data)
                    data_batch_pad.append(data_pad)

                batch = Batch.from_data_list(data_batch, follow_batch = [])
                batches.append(batch)

                batch_out = Batch.from_data_list(data_batch_out, follow_batch = [])
                batches_out.append(batch_out)

                batch_pad = Batch.from_data_list(data_batch_pad, follow_batch = [])
                batches_pad.append(batch_pad)

                batch_out_pad = Batch.from_data_list(data_batch_out_pad, follow_batch = [])
                batches_out_pad.append(batch_out_pad)
                
                originals.append(img_batch)
                
                if self.merging:
                    # Create merged input + output graphs
                    # for each index, replace the output by empty output to force it to learn generating them
                    # concatenated in a batch
                    data_merged = []
                    data_merged_out = []
                    for i in range(len(task["train"])):
                        for idx, t in enumerate(task["train"]):
                            merged = sample_to_graph_merged(t["input"], t["output"],
                                                             empty_target=False,
                                                             pad_value = self.pad_value)
                            data_merged_out.append(merged)
                            
                            if i != idx:
                                data_merged.append(merged)
                            else:
                                merged = sample_to_graph_merged(t["input"], t["output"],
                                                                 empty_target=True,
                                                                 pad_value = self.pad_value)
                                data_merged.append(merged)
                                
                    data_merged = Batch.from_data_list(data_merged, follow_batch = [])
                    data_merged_out = Batch.from_data_list(data_merged_out, follow_batch = [])
                    batches_merged.append(data_merged)
                    batches_merged_out.append(data_merged_out)

                #torch.save(batch, graph_file)
                nb += 1
        self.batches = batches
        self.batches_out = batches_out
        
        self.batches_pad = batches_pad
        self.batches_out_pad = batches_out_pad
        
        self.batches_merged = batches_merged
        self.batches_merged_out = batches_merged_out
        
        self.originals = originals

        print(f"Read {nb} files")
 
    def len(self):
        return len(self.batches)

    def get(self, idx):
        if not self.merging:
            return (self.batches[idx], self.batches_out[idx],
                    self.batches_pad[idx], self.batches_out_pad[idx],
                    self.originals[idx])
        else:
            return (self.batches[idx], self.batches_out[idx],
                    self.batches_pad[idx], self.batches_out_pad[idx],
                    self.batches_merged[idx], self.batches_merged_out[idx],
                    self.originals[idx])
        
ds = ARCGeometricDataset(training_path, "./geometric", pad_value=10, merging=True)
draw_batch_images(ds[1], "1", use_pad=True, merging = True)
draw_batch_images(ds[0], "0", use_pad=True, merging = True)
draw_batch_images(ds[338], "338", use_pad=True, merging=True)

        
# Customized        
class ARCCollater(object):
    def __init__(self, follow_batch, merging: bool = False):
        self.follow_batch = follow_batch
        self.merging = merging

    def collate(self, batch):
        #res = []
        batches_out = []
        batches_out_pad = []
        batches = []
        batches_pad = []
        batches_merged = []
        batches_merged_out = []
        images = []
        for e in batch:
            if not self.merging:
                batch_data, batch_out_data, batch_pad, batch_out_pad, imgs = e
            else:
                batch_data, batch_out_data, batch_pad, batch_out_pad, batch_merged, batch_merged_out, imgs = e
                batches_merged.append(batch_merged)
                batches_merged_out.append(batch_merged_out.x)
            batches.append(batch_data)
            batches_pad.append(batch_pad)
            batches_out.append(batch_out_data.x)
            batches_out_pad.append(batch_out_pad.x)
            images.append(imgs)

        batches_out = torch.cat(batches_out, dim=0).long().squeeze()
        batches_out_pad = torch.cat(batches_out_pad, dim=0).long().squeeze()
            
        if not self.merging:
            return (batches, batches_out, batches_pad, batches_out_pad, images)
        else:
            batches_merged_out = torch.cat(batches_merged_out, dim=0).long().squeeze()
            return (batches, batches_out, batches_pad, batches_out_pad, batches_merged, batches_merged_out, images)

    def __call__(self, batch):
        return self.collate(batch)


class ARCDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], merging: bool = True,
                 **kwargs):
        super(ARCDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=ARCCollater(follow_batch, merging), **kwargs)
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric import utils


class FocalLoss(nn.Module):

    """Focal Loss."""

    def __init__(self, alpha=1, gamma=2, reduce=True):
        """Constructor."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        """Forward propagation."""
        BCE_loss = self.ce_loss_fn(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # type: ignore

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class Net(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        print("hparams", hparams)
        super(Net, self).__init__()
        
        self.data_path = Path(hparams.data_path)
        self.training_path = self.data_path / 'training'
        self.evaluation_path = self.data_path / 'evaluation'
        self.test_path = self.data_path / 'test'
        self.use_pad = hparams.use_pad
        self.merging = hparams.merging
        self.num_classes = hparams.num_classes
        self.lr = hparams.lr
        self.focal_alpha = hparams.focal_alpha
        self.focal_lambda = hparams.focal_lambda
        
        self.conv1 = GCNConv(hparams.num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
#        self.conv3 = GCNConv(64, 32)
#        self.conv4 = GCNConv(128, 256)
#        self.conv5 = GCNConv(256, 128)
#        self.conv6 = GCNConv(128, 64)
#        self.conv7 = GCNConv(64, 32)
        self.conv_last = GCNConv(32, 11)
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_lambda)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
#        x = self.conv3(x, edge_index)
#        x = F.relu(x)
#        x = F.dropout(x, training=self.training)
#        x = self.conv4(x, edge_index)
#        x = F.relu(x)
#        x = F.dropout(x, training=self.training)
#        x = self.conv5(x, edge_index)
#        x = F.relu(x)
#        x = F.dropout(x, training=self.training)
#        x = self.conv6(x, edge_index)
#        x = F.relu(x)
#        x = F.dropout(x, training=self.training)
#        x = self.conv7(x, edge_index)
#        x = F.relu(x)
#        x = F.dropout(x, training=self.training)
        x = self.conv_last(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        data_batches = []
        images = []
        if not self.merging:
            if self.use_pad:
                _, _, data_batch, data_target, _ = batch
            else:
                data_batch, data_target, _, _, _ = batch
        else:
            _, _, _, _, data_batch, data_target, _ = batch
        out = []
        for db in data_batch:
            out.append(self(db))
        out = torch.cat(out, dim=0)
        loss = self.loss_fn(out, data_target)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data_batches = []
        images = []
        if not self.merging:
            if self.use_pad:
                _, _, data_batch, data_target, images = batch
            else:
                data_batch, data_target, _, _, images = batch
        else:
            _, _, _, _, data_batch, data_target, images = batch
        out = []
        for db in data_batch:
            db.to(device, non_blocking=True)
            out.append(self(db))
        out = torch.cat(out, dim=0)
        loss = self.loss_fn(out, data_target)
        out = torch.argmax(out, dim=1)
        tp = utils.true_positive(out, data_target, self.num_classes-1)
        tn = utils.true_negative(out, data_target, self.num_classes-1)
        fp = utils.false_positive(out, data_target, self.num_classes-1)
        fn = utils.false_negative(out, data_target, self.num_classes-1)

        return {'val_loss': loss, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tps = torch.sum(torch.cat([x['tp'] for x in outputs])).to(torch.float)
        tns = torch.sum(torch.cat([x['tn'] for x in outputs])).to(torch.float)
        fps = torch.sum(torch.cat([x['fp'] for x in outputs])).to(torch.float)
        fns = torch.sum(torch.cat([x['fn'] for x in outputs])).to(torch.float)
        
        precision = tps / (tps + fps)
        precision[torch.isnan(precision)] = 0
        
        recall = tps / (tps + fns)
        recall[torch.isnan(recall)] = 0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score[torch.isnan(f1_score)] = 0
    
        tensorboard_logs = {'val_loss': avg_loss, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        
    def train_dataloader(self):
        ds = ARCGeometricDataset(self.training_path, "./geometric", merging=self.merging, pad_value=10)
        
        train_dataloader = ARCDataLoader(ds, batch_size=1, shuffle=True, num_workers=5, merging=self.merging)
        return train_dataloader

    def val_dataloader(self):
        ds = ARCGeometricDataset(self.evaluation_path, "./geometric", merging=self.merging, pad_value=10)
        
        val_dataloader = ARCDataLoader(ds, batch_size=1, shuffle=False, num_workers=5, merging=self.merging)
        return val_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--use_pad", action="store_true", help="Use graph padding.")
        parser.add_argument("--merging", action="store_true", help="Use graph merging.")
        parser.add_argument('--data_path', type=str)
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--num_node_features', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--focal_alpha', type=float, default=0.5)
        parser.add_argument('--focal_lambda', type=float, default=2.0)
        return parser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

#, use_pad: bool = True, merging: bool = True,
#num_classes=10, num_node_features: int = 1,
#                 data_path = Path('kaggle/input/abstraction-and-reasoning-challenge/')
                                  
args = [
    "--use_pad",
    "--merging",
    "--num_classes", "10",
    "--num_node_features", "1",
    "--data_path", "kaggle/input/abstraction-and-reasoning-challenge/",
    "--lr", "0.000001",
    "--focal_alpha", "0.5",
    "--focal_lambda", "2.0",
]

parser = ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser = Net.add_model_specific_args(parser)
hparams = parser.parse_args(args)

model = Net(hparams)

exp = "arc_geometric_v7"
wandb_logger = WandbLogger(project="mandubian-arc-graph", name=exp)
#logger = TensorBoardLogger("tb_logs", name=exp)


# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=f"./checkpoints/{exp}",
    save_top_k=True,
    verbose=False,
    monitor='val_loss',
    mode='min',
    #prefix=exp + "_"
)

trainer = Trainer(max_epochs=400, logger=wandb_logger, pin_memory=True,
                  distributed_backend=None, checkpoint_callback=checkpoint_callback, gpus=1)
import wandb
wandb.init()
trainer.fit(model)
best_model = Net.load_from_checkpoint(checkpoint_path="./checkpoints/arc_geometric_v1_epoch=99.ckpt")
best_model.freeze()
best_model
submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
display(submission.head())
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred
for output_id in submission.index:
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
    print("reading file", f)
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    # skipping over the training examples, since this will be naive predictions
    # we will use the test input grid as the base, and make some modifications
    # print("task", task)
    data = task['test'][pair_id]['input'] # test pair input
    
    data_merged = []
    output_size = 0
    task_test = task['test'][pair_id] # test task
    test_height_input = len(task_test["input"])
    test_width_input = len(task_test["input"][0])
    min_height_diff = 1e20
    ref_output = None
    ref_output_height = None
    ref_output_width = None
    ref_output_max = None
    ref_output_rate = None
    ref_rate_diff = 1e10
    for idx, t in enumerate(task["train"]):
        t_height_input = len(t["input"])
        t_width_input = len(t["input"][0])
        t_height_output = len(t["output"])
        t_width_output = len(t["output"][0])
        # engineering output size here ;)
        if test_height_input == t_height_input and test_width_input == t_width_input:
            ref_output = t["output"]
        elif test_height_input == t_height_input:
            ref_output_height = t["output"]
        elif test_width_input == t_width_input:
            ref_output_width = t["output"]
        else:
            r = t_height_input / t_width_input - test_height_input / test_width_input
            if r < ref_rate_diff:
                ref_rate_diff = r
                ref_output_rate = t["output"]
        
        merged = sample_to_graph_merged(t["input"], t["output"],
                                        empty_target=False,
                                        pad_value = 10)
        data_merged.append(merged)

    ref_output = ref_output or ref_output_height or ref_output_width or ref_output_rate
    ref_height_output = len(ref_output)
    ref_width_output = len(ref_output[0])
    test_merged = sample_to_graph_merged(task_test["input"], ref_output,
                                    empty_target=True,
                                    pad_value = 10)
    data_merged.append(test_merged)
    
    batch = Batch.from_data_list(data_merged)
    
    #print("batch", batch)
    output = best_model(batch)
    #print("output", output)
    output = torch.nn.functional.softmax(output)
    #print("output", output)
    output = torch.argmax(output, dim=1)
    output = output[:-ref_height_output * -ref_width_output]
    output = output.view(ref_height_output, ref_width_output)
    output = output.cpu().numpy().tolist()
    pred_1 = flattener(output)
    pred = pred_1
    submission.loc[output_id, 'output'] = pred
    #graph = sample_to_graph(data)
    # for the first guess, predict that output is unchanged
    #pred_1 = flattener(data)
    # for the second guess, change all 0s to 5s
    #data = [[5 if i==0 else i for i in j] for j in data]
    #pred_2 = flattener(data)
    # for the last gues, change everything to 0
    #data = [[0 for i in j] for j in data]
    #pred_3 = flattener(data)
    # concatenate and add to the submission output
    #pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    #submission.loc[output_id, 'output'] = pred
print(submission)
submission.to_csv('submission.csv')
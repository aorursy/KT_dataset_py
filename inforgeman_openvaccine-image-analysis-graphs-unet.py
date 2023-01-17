import graphviz
def gv(s): return graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')
gv(''' node [shape="record" style="rounded,filled" fillcolor=gold];
"test.json"->"sequence: ACGFUG";"test.json"->"structure: ...((..)).."-> "RNA graph";"sequence: ACGFUG" -> "RNA graph"-> "{id}_x.png";
"RNA graph"-> "placement of elements in px";"placement of elements in px"->"{id}.json"->"color_to_value"
"test.json"->"reactivity: [...]";"reactivity: [...]"->"value_to_color"->"RNA graph/reactivity"
"RNA graph/reactivity"->"{id}_y.png"
"{id}_y.png"->"U_NET";"{id}_x.png"->"U_NET"->"prediction.png"->"color_to_value"->"results: [...]"
''')
path_train = '../input/stanford-covid-vaccine/train.json'
path_test = '../input/stanford-covid-vaccine/test.json'

import json
train_data = [json.loads(json_line) for json_line in open(path_train).readlines() if '"SN_filter":1.0' in json_line]
test_data = [json.loads(json_line) for json_line in open(path_test).readlines()]
# Adding RiboGraphViz, pygraphviz
!mkdir tmp;cd /kaggle/working/tmp; git clone https://github.com/DasLab/RiboGraphViz.git;cp /kaggle/working/tmp/RiboGraphViz/RiboGraphViz /kaggle/working/ -r
!yes|conda install -c anaconda pygraphviz
from RiboGraphViz import RGV

def make_x_graph_simple(seq, struct):
    map_color = {'G': 'r', 'A': 'b', 'U': 'g', 'C': 'y'}
    seq_color = ''.join([map_color[x] for x in seq[:68]] + ['k' for x in seq[68:]])
    alpha = [[1, 0.2][i > 67] for i, _ in enumerate(seq)]
    rgv = RGV(struct)
    rgv.draw(align=True, c=seq_color, align_mode='end', alpha=alpha,)

make_x_graph_simple(train_data[88]['sequence'], train_data[88]['structure'])
gv(''' node [shape="record" style="rounded,filled" fillcolor=gold];
"my_neighbour 4"->"42";"my_neighbour 5"->"42";"me 2"->"42"
"my_neighbour 1"->"142";"my_neighbour 2"->"142";"my_neighbour 3"->"142";"me 1"->"142"
''')
def split_chain(sg):
    stack = []
    l = []
    r = []
    counter = 0
    current_char = sg[0]
    for i, x in enumerate(sg):
        if x == '.':
            l.append(-1)
        if x == '(':
            l.append(-1)
            stack.append(i)
        if x == ')':
            l.append(stack.pop())

    for i in range(len(sg) - 1, -1, -1):
        x = sg[i]
        if x == '.':
            r.append(-1)
        if x == ')':
            r.append(-1)
            stack.append(i)
        if x == '(':
            r.append(stack.pop())

    r = r[::-1]
    all = []
    for i, _ in enumerate(l):
        if l[i] != -1:
            result = l[i]
        elif r[i] != -1:
            result = r[i]
        else:
            result = -1
        all.append(result)

    indices = []
    current_char = sg[0]
    chunk_start = 0

    for i, x in enumerate(sg):
        if x != current_char:
            current_char = x
            indices.append([chunk_start, i])
            chunk_start = i
        if current_char == '(' or current_char == ')':
            if all[i-1] - all[i] > 1:
                indices.append([chunk_start, i])
                chunk_start = i
    else:
        indices.append([chunk_start, len(sg)])
    return indices

def get_parallel_chunk(chunks):
    stack = []
    branches = []
    for i, x in enumerate(chunks):
        if x['type'] == '(':
            stack.append(i)
        if x['type'] == ')':
            branches.append([stack.pop(), i])

    xxx = []
    for i, x in enumerate(chunks):
        for y in branches:
            if i in y:
                z = set(y) - {i}
                xxx.append(next(iter(z)))

    mmm = []
    for i in xxx:
        mmm.append(chunks[i]['value'][::-1])

    all_str = ''.join(mmm)
    c = 0
    result = []
    m = [x['value'] for x in chunks if x['type'] in ['(', ')']]
    for x in m:
        result.append(all_str[c : c + len(x)])
        c = c + len(x)
    counter = 0
    for chunk in chunks:
        if chunk['type'] in ['(', ')']:
            chunk['third'] = result[counter]
            counter = counter + 1
        else:
            chunk['third'] = '-'
    return chunks

def get_chunks(sequence, indices, chain):
    return [{'value': sequence[x[0] : x[1]], 'type': chain[x[0]]} for x in indices]
def make_x_graph(seq, struct):
    chunks = get_parallel_chunk(get_chunks(seq, split_chain(struct), struct))
    thirds = ''.join(['-'*len(x['value']) if x['third']=='-' or x['third']=='' else x['third']  for x in chunks])
    thirds = [{'G':1, 'A':2, 'C':3, 'U':4, '-':0}[x] for x in thirds]
    seq = [{'G':1, 'A':2, 'C':3, 'U':4}[x] for x in seq]
    seq_3neighbors = [[([0]+seq+[0])[i+j] for j in range(-1,2)] for i in range(1, len(seq) + 1)]
    seq_4neighbors = [[thirds[i]]+x for i, x in enumerate(seq_3neighbors)]
    seq_4_to_1 = [[int(''.join(map(str, x)), 5) / 624 + 1, int(''.join(map(str, x)), 5) / 124][x[0] == 0] for x in seq_4neighbors]
    alpha = [[1, 0.05][i > 67] for i, _ in enumerate(seq)]
    rgv = RGV(struct)
    rgv.draw(align=True, c=seq_4_to_1, cmap='gist_rainbow', align_mode='end', alpha=alpha,)


make_x_graph_simple(train_data[2]['sequence'], train_data[2]['structure'])
make_x_graph(train_data[2]['sequence'], train_data[2]['structure'])
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

def get_xy(self,): # stolen from RGV.draw
    N = len(self.secstruct)
    plot_nodes = [n for n in list(self.G.nodes) if isinstance(n, str)]
    subgraph = self.G.subgraph(plot_nodes)
    subgraph = subgraph.to_undirected()

    if not self.is_multi:
        if 'n0' in list(subgraph.nodes()):
            subgraph.add_edge('n0', "5'", len=2)
        else:
            subgraph.add_edge('h1b', "5'", len=2)

        if 'n%d' % (N - 1) in list(subgraph.nodes()):
            subgraph.add_edge('n%d' % (N - 1), "3'", len=2)

    pos = graphviz_layout(subgraph, prog='neato')

    if not self.is_multi:
        fiveprime_x, fiveprime_y = pos["5'"]

        if "3'" in list(subgraph.nodes()):
            threeprime_x, threeprime_y = pos["3'"]

    for (u, v) in list(subgraph.edges()):
        if u.startswith('n') and v.startswith('n'):
            x1, x2, y1, y2 = pos[u][0], pos[v][0], pos[u][1], pos[v][1]
            break

    bond_width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 0.8
    node_positions_x, node_positions_y = {}, {}

    for node in list(subgraph.nodes()):
        if node.startswith('n'):
            seq_ind = int(node[1:])
            node_positions_x[seq_ind] = pos[node][0]
            node_positions_y[seq_ind] = pos[node][1]

    for i, stem in enumerate(self.stems):

        start_x, start_y = pos['h%da' % (i + 1)]
        fin_x, fin_y = pos['h%db' % (i + 1)]

        x_dist = fin_x - start_x
        y_dist = fin_y - start_y

        path_length = np.sqrt((start_x - fin_x) ** 2 + (start_y - fin_y) ** 2)
        stem_angle = np.arctan2(y_dist, x_dist)

        x_diff = np.cos(stem_angle + np.pi / 2) * 0.5 * bond_width * self.helix_width
        y_diff = np.sin(stem_angle + np.pi / 2) * 0.5 * bond_width * self.helix_width

        for j in range(len(stem)):
            x_midpoint = start_x + j * x_dist / (len(stem) - 0.99)
            y_midpoint = start_y + j * y_dist / (len(stem) - 0.99)

            if stem_angle < 0:
                node_positions_x[stem[j][1]] = x_midpoint + x_diff
                node_positions_x[stem[j][0]] = x_midpoint - x_diff
                node_positions_y[stem[j][1]] = y_midpoint + y_diff
                node_positions_y[stem[j][0]] = y_midpoint - y_diff

            else:
                node_positions_x[stem[j][0]] = x_midpoint + x_diff
                node_positions_x[stem[j][1]] = x_midpoint - x_diff
                node_positions_y[stem[j][0]] = y_midpoint + y_diff
                node_positions_y[stem[j][1]] = y_midpoint - y_diff

    node_pos_list_x = [node_positions_x[i] - node_positions_x[0] for i in range(N)]
    node_pos_list_y = [node_positions_y[i] - node_positions_y[0] for i in range(N)]

    vec_01_x = node_pos_list_x[N - 1]
    vec_01_y = node_pos_list_y[N - 1]

    curr_angle = np.arctan2(vec_01_y, vec_01_x)

    new_node_pos_list_x = [
        np.cos(-1 * curr_angle) * node_pos_list_x[i]
        - np.sin(-1 * curr_angle) * node_pos_list_y[i]
        for i in range(N)
    ]
    new_node_pos_list_y = [
        np.sin(-1 * curr_angle) * node_pos_list_x[i]
        + np.cos(-1 * curr_angle) * node_pos_list_y[i]
        for i in range(N)
    ]

    if np.mean(new_node_pos_list_y) < 0:
        new_node_pos_list_y = [-1 * y for y in new_node_pos_list_y]
    if new_node_pos_list_y[1] < new_node_pos_list_y[0]:
        new_node_pos_list_y = [-1 * y for y in new_node_pos_list_y]

    return zip(new_node_pos_list_x, new_node_pos_list_y)

RGV.get_xy = get_xy
def make_y_graph(target, struct):
    alpha = [[1, 0.2][i > 67] for i, _ in enumerate(struct)]
    target = target + [0 for x in range(len(struct) - 68)]
    rgv = RGV(struct)
    items_xy = rgv.get_xy()
    ax = rgv.draw(align=True, c=target, cmap='gist_rainbow', align_mode='end', alpha=alpha,)
    for x, y in items_xy:
        ax.plot(x, y, 'b+')

make_y_graph(train_data[0]['reactivity'], train_data[0]['structure'])
import matplotlib.pyplot as plt

def save_x_graph(seq, struct, cut_i, path):
    chunks = get_parallel_chunk(get_chunks(seq, split_chain(struct), struct))
    thirds = ''.join(['-'*len(x['value']) if x['third']=='-' or x['third']=='' else x['third']  for x in chunks])
    thirds = [{'G':1, 'A':2, 'C':3, 'U':4, '-':0}[x] for x in thirds]
    seq = [{'G':1, 'A':2, 'C':3, 'U':4}[x] for x in seq]
    seq_3neighbors = [[([0]+seq+[0])[i+j] for j in range(-1,2)] for i in range(1, len(seq) + 1)]
    seq_3neighbors = [sorted([x[0],x[2]])+[x[1]] for x in seq_3neighbors]
    seq_4neighbors = [[thirds[i]]+x for i, x in enumerate(seq_3neighbors)]
    seq_4_to_1 = [[int(''.join(map(str, x)), 5) / 624 + 1, int(''.join(map(str, x)), 5) / 124][x[0] == 0] for x in seq_4neighbors]
    
    alpha = [[1, 0][i > cut_i-1] for i, _ in enumerate(seq)]
    rgv = RGV(struct)
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.clf()
    ax = rgv.draw(align=True, c=seq_4_to_1, cmap='gist_rainbow', align_mode='end', alpha=alpha,)
    xy_plot_coords = list(rgv.get_xy())[:cut_i]
    fig.canvas.draw()
    xy_pixels = ax.transData.transform(xy_plot_coords)
    _, height = fig.canvas.get_width_height()
    final_pixel_coords = [(int(round(xy[0])), int(round(height - xy[1]))) for xy in xy_pixels]
    plt.savefig(path, dpi=70)
    plt.close('all')
    return final_pixel_coords
import pandas as pd
df = pd.DataFrame()
min_max={}
for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    z = []
    for x in train_data:
        z.extend(x[y_type])
    df.insert(0, y_type, z, True)
    min_max[y_type] = [df[y_type].min(), df[y_type].max()]

df.plot.hist(bins=40, alpha=0.5)
min_max
for x in min_max:
    min_max[x][1] = min(min_max[x][1], 2.2)
min_max
# checking colormap scaling

import matplotlib.colors as colors
from matplotlib import cm
cm_rainbow = plt.get_cmap('gist_rainbow')
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

_, axs = plt.subplots(1,3, figsize=(19, 1))


def scale_cmap(cmap, min_v, max_v):
    return colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{min_v:.3f},{max_v:.3f})', cmap(np.linspace(min_v, max_v, 256)))

axs[0].imshow(gradient, aspect=10, cmap = cm_rainbow)
axs[1].imshow(gradient, aspect=10, cmap = scale_cmap(cm_rainbow, 0.1, 0.4))
axs[2].imshow(gradient, aspect=10, cmap = scale_cmap(cm_rainbow, 0.6, 0.9))
for i in range(3):
    axs[i].set_title(['rainbow', 'rainbow (0.1-0.4)', 'rainbow (0.6-0.9)'][i])
def save_y_graph(target, struct, path, minmax):
    y_min, y_max = minmax
    target = [max(y_min, min(x, y_max)) for x in target]
    cm_scaled = scale_cmap(cm_rainbow, (min(target) - y_min) / (y_max - y_min), (max(target)- y_min)/ (y_max - y_min))
    targ_len = len(target)
    alpha = [[1, 0][i >= targ_len] for i, _ in enumerate(struct)]
    target = target + [0 for x in range(len(struct) - targ_len)]
    rgv = RGV(struct)
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.clf()
    rgv.draw(align=True, c=target, cmap=cm_scaled, align_mode='end', alpha=alpha,)
    plt.savefig(path, dpi=70)
    plt.close('all')
!mkdir imgs;cd imgs;mkdir input_train;mkdir output_reactivity;mkdir output_deg_Mg_pH10;mkdir output_deg_Mg_50C;mkdir xy_data

from fastprogress.fastprogress import progress_bar

input_train_path = '/kaggle/working/imgs/input_train'
output_path = '/kaggle/working/imgs/output'
xy_path = '/kaggle/working/imgs/xy_data'

for i in progress_bar(range(len(train_data))):
    data_dict = train_data[i]
    seq_id = data_dict['id']
    sequence = data_dict['sequence']
    structure = data_dict['structure']

    xy_pixels = save_x_graph(sequence, structure, 68, f'{input_train_path}/{seq_id}.png')
    with open(f'{xy_path}/{seq_id}.json', 'w') as f: json.dump({'x':[x for x,_ in xy_pixels], 'y':[y for _,y in xy_pixels]}, f)
        
    for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
        save_y_graph(data_dict[y_type], structure, f'{output_path}_{y_type}/{seq_id}.png', min_max[y_type])   
#testing random sequence
import random
from pathlib import Path
import cv2

file = random.choice(list(Path(input_train_path).iterdir()))
with open(f'{xy_path}/{file.stem}.json') as json_file: data = json.load(json_file)


png_paths = [f'{input_train_path}/{file.stem}.png'] + [f'{output_path}_{y_type}/{file.stem}.png' for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']]

fig, axs = plt.subplots(1,4,figsize=(33, 33))
for i, ax in enumerate(axs):
    img = cv2.cvtColor(cv2.imread(png_paths[i]), cv2.COLOR_BGR2RGBA)
    for x, y in zip(data['x'], data['y']):
        img = cv2.circle(img, (x, y), radius=0, color=(0, 0, 255), thickness=3)
    ax.imshow(img, interpolation='nearest')
    ax.set_title(Path(png_paths[i]).parts[-2])
import matplotlib
def get_values_from_image(img, pixel_coords, minmax):
    def get_value_from_cm(color, cmap, colrange):
        color = np.array(color) / 255.0
        r = np.linspace(colrange[0], colrange[1], 256)
        norm = matplotlib.colors.Normalize(colrange[0], colrange[1])
        mapvals = cmap(norm(r))[:, :4]
        distance = np.sum((mapvals - color) ** 2, axis=1)
        return r[np.argmin(distance)]

    result = []
    for x, y in pixel_coords:
        color = img[y, x]
        decoded_value = get_value_from_cm(color, cm.gist_rainbow, colrange=minmax)
        result.append(decoded_value)
    return result


errors = {y_type: [] for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']}
for sample in train_data:
    seq_id = sample['id']
    with open(f'{xy_path}/{seq_id}.json') as json_file: data = json.load(json_file)
    for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
        img = cv2.cvtColor(cv2.imread(f'{output_path}_{y_type}/{seq_id}.png'), cv2.COLOR_BGR2RGBA)       
        errors[y_type].append((np.mean(np.abs(np.array(sample[y_type]) - np.array(get_values_from_image(img, zip(data['x'], data['y']), min_max[y_type]))))))

for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    print(f'{y_type} MAE {np.mean(np.array(errors[y_type]))}')
!yes|pip install fastai==1.0.61

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
src = ImageImageList.from_folder(input_train_path).split_by_rand_pct(0.1, seed=13)

tfms = get_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=0,
    max_zoom=1,
    max_lighting=None,
    max_warp=0.0,
    p_affine=1.0,
)


def get_data(bs, y_type):
    data = (
        src.label_from_func(lambda x: Path(output_path + '_' + y_type) / x.name)
        .transform(tfms, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats, do_y=True)
    )

    data.c = 3
    return data
# proudly stolen from fastai course3

def gram_matrix(x): 
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = (
            ['pixel',]
            + [f'feat_{i}' for i in range(len(layer_ids))]
            + [f'gram_{i}' for i in range(len(layer_ids))]
        )

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [
            base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.feat_losses += [
            base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


base_loss = F.l1_loss
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])
data = get_data(4, 'reactivity')
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(19, 19))
lr = 1e-3
bs =  4
wd = 1e-3
arch = models.resnet34

def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(15, lrs, pct_start=pct_start)
    learn.save(save_name)


for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    data = get_data(bs, y_type)
    learn = unet_learner(data,arch,wd=wd,bottle=True,loss_func=feat_loss,callback_fns=LossMetrics,blur=True,norm_type=NormType.Weight,).to_fp16(loss_scale=1)
    do_fit(y_type, slice(lr * 10))
    learn.unfreeze()
    do_fit(y_type, slice(1e-5, lr))
for model in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    learn.load(model)
    learn.show_results(rows=2, imgsize=17)
#simple checking
learn.load('reactivity')
num = 22
seq_id = train_data[num]['id']
file = f'{input_train_path}/{seq_id}.png'
with open(f'{xy_path}/{seq_id}.json') as json_file: data = json.load(json_file)

img = open_image(file)
_,img_pred,_ = learn.predict(img)

img = cv2.cvtColor(image2np(fastai.vision.Image(img_pred).px), cv2.COLOR_RGB2RGBA)
fig, ax = plt.subplots(figsize=(9, 9))
for x, y in zip(data['x'], data['y']):
    img = cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
ax.imshow(img, interpolation='nearest')

sum_ = 0
img_pred = cv2.cvtColor(image2np(fastai.vision.Image(img_pred).px)*255, cv2.COLOR_RGB2RGBA)
for x,y in zip(train_data[num]['reactivity'], get_values_from_image(img_pred, zip(data['x'], data['y']), min_max['reactivity'])):
    sum_ = sum_ + abs(x - y)
sum_/68
#check all train_data
errors = {y_type: [] for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']}
for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    learn.load(y_type)
    for i in progress_bar(range(len(train_data))):
        sample = train_data[i]
        seq_id = sample['id']
        with open(f'{xy_path}/{seq_id}.json') as json_file: data = json.load(json_file)
        file = f'{input_train_path}/{seq_id}.png'
        img = open_image(file)
        _,img_pred,_ = learn.predict(img)
        img_pred = cv2.cvtColor(image2np(fastai.vision.Image(img_pred).px)*255, cv2.COLOR_RGB2RGBA)
        errors[y_type].append((np.mean(np.abs(np.array(sample[y_type]) - np.array(get_values_from_image(img_pred, zip(data['x'], data['y']), min_max[y_type]))))))

for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    print(f'{y_type} MAE {np.mean(np.array(errors[y_type]))}')
!cd imgs;mkdir input_test

input_test_path = '/kaggle/working/imgs/input_test'

for i in progress_bar(range(len(test_data))):
    data_dict = test_data[i]
    seq_id = data_dict['id']
    sequence = data_dict['sequence']
    structure = data_dict['structure']

    xy_pixels = save_x_graph(sequence, structure, len(sequence),f'{input_test_path}/{seq_id}.png')   
    with open(f'{xy_path}/{seq_id}.json', 'w') as f: json.dump({'x':[x for x,_ in xy_pixels], 'y':[y for _,y in xy_pixels]}, f)
#testing random sequence
file = random.choice(list(Path(input_test_path).iterdir()))
with open(f'{xy_path}/{file.stem}.json') as json_file: data = json.load(json_file)

fig, ax = plt.subplots(figsize=(11, 11))
img = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGBA)
for x, y in zip(data['x'], data['y']):
    img = cv2.circle(img, (x, y), radius=0, color=(0, 0, 255), thickness=3)
ax.imshow(img, interpolation='nearest')
results_dict = {}
for y_type in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:
    learn.load(y_type)
    for i in progress_bar(range(len(test_data))):
        sample = test_data[i]
        seq_id = sample['id']
        with open(f'{xy_path}/{seq_id}.json') as json_file: data = json.load(json_file)
        file = f'{input_test_path}/{seq_id}.png'
        img = open_image(file)
        _,img_pred,_ = learn.predict(img)
        img_pred = cv2.cvtColor(image2np(fastai.vision.Image(img_pred).px)*255, cv2.COLOR_RGB2RGBA)
        if seq_id not in results_dict:
            results_dict[seq_id] = {}
        results_dict[seq_id][y_type]=get_values_from_image(img_pred, zip(data['x'], data['y']), min_max[y_type])        
table=[]
for i in progress_bar(range(len(test_data))):
    sample = test_data[i]
    seq_id = sample['id']
    results = results_dict[seq_id]
    seq_ids = [f'{seq_id}_{i}'for i in range(0, sample['seq_length'])]
    for row in zip(seq_ids,results['reactivity'], results['deg_Mg_pH10'], results['deg_Mg_50C']):
        table.append(list(row)+[0, 0])

df = pd.DataFrame(table, columns=['id_seqpos','reactivity','deg_Mg_pH10','deg_Mg_50C','deg_pH10','deg_50C'])
df.to_csv('submission.csv', index=False)

import gzip, shutil
with open('submission.csv', 'rb') as f_in:
    with gzip.open('submission.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
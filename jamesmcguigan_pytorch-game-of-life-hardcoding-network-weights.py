from __future__ import annotations



import math

import os

import re

from abc import ABCMeta

from typing import List

from typing import TypeVar

from typing import Union



import humanize

import numpy as np

import torch

import torch.nn as nn



device   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

__file__ = './notebook.py'  # hardcode for jupyer notebook 



# Source: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

def batch(iterable, n=1):

    l = len(iterable)

    for ndx in range(0, l, n):

        yield iterable[ndx:min(ndx + n, l)]





# noinspection PyTypeChecker

T = TypeVar('T', bound='GameOfLifeBase')

class GameOfLifeBase(nn.Module, metaclass=ABCMeta):

    """

    Base class for GameOfLife based NNs

    Handles: save/autoload, freeze/unfreeze, casting between data formats, and training loop functions

    """

    def __init__(self):

        super().__init__()

        self.loaded    = False  # can't call sell.load() in constructor, as weights/layers have not been defined yet

        self.device    = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.criterion = nn.MSELoss()





    def weights_init(self, layer):

        ### Default initialization seems to work best, at least for Z shaped ReLU1 - see GameOfLifeHardcodedReLU1_21.py

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):

            ### kaiming_normal_ corrects for mean and std of the relu function

            ### xavier_normal_ works better for ReLU6 and Z shaped activations

            if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU)):

                nn.init.kaiming_normal_(layer.weight)

                # nn.init.xavier_normal_(layer.weight)

                if layer.bias is not None:

                    # small positive bias so that all nodes are initialized

                    nn.init.constant_(layer.bias, 0.1)

        else:

            # Use default initialization

            pass







    ### Prediction



    def __call__(self, *args, **kwargs) -> torch.Tensor:

        if not self.loaded: self.load()  # autoload on first function call

        return super().__call__(*args, **kwargs)



    def predict(self, inputs: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> np.ndarray:

        """

        Wrapper function around __call__() that returns a numpy int8 array for external usage

        Will auto-detect the largest batch size capable of fitting into GPU memory

        Output is squeezed, so will work both for single board, as well as a batch

        """

        inputs     = self.cast_inputs(inputs)  # cast to 4D tensor

        batch_size = len(inputs)               # keep halving batch_size until it is small enough to fit in CUDA memory

        while True:

            try:

                outputs = []

                for input in batch(inputs, batch_size):

                    output = self(input)

                    output = self.cast_int(output).detach().cpu().numpy()

                    outputs.append(output)

                outputs = np.concatenate(outputs).squeeze()

                return outputs

            except RuntimeError as exception:  # CUDA out of memory exception

                torch.cuda.empty_cache()

                if batch_size == 1: raise exception

                batch_size = math.ceil( batch_size / 2 )







    ### Training



    def loss(self, outputs, expected, input):

        return self.criterion(outputs, expected)



    def accuracy(self, outputs, expected, inputs) -> float:

        # noinspection PyTypeChecker

        return torch.sum(self.cast_int(outputs) == self.cast_int(expected)).cpu().numpy() / np.prod(outputs.shape)







    ### Freeze / Unfreeze



    def freeze(self: T) -> T:

        if not self.loaded: self.load()

        for name, parameter in self.named_parameters():

            parameter.requires_grad = False

        return self



    def unfreeze(self: T) -> T:

        if not self.loaded: self.load()

        for name, parameter in self.named_parameters():

            parameter.requires_grad = True

        return self







    ### Load / Save Functionality



    @property

    def filename(self) -> str:

        return os.path.join( os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth')





    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html

    def save(self: T) -> T:

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        torch.save(self.state_dict(), self.filename)

        print(f'{self.__class__.__name__}.savefile(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')

        return self





    def load(self: T, load_weights=True) -> T:

        ### Disable loading of weights for Notebook Demo

        # if load_weights and os.path.exists(self.filename):

        #     try:

        #         self.load_state_dict(torch.load(self.filename))

        #         print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')

        #     except Exception as exception:

        #         # Ignore errors caused by model size mismatch

        #         print(f'{self.__class__.__name__}.load(): model has changed dimensions, reinitializing weights\n')

        #         self.apply(self.weights_init)

        # else:

        #     if load_weights: print(f'{self.__class__.__name__}.load(): model file not found, reinitializing weights\n')

        #     else:            print(f'{self.__class__.__name__}.load(): reinitializing weights\n')

        #     self.apply(self.weights_init)



        self.loaded = True    # prevent any infinite if self.loaded loops

        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU

        self.eval()           # default to production mode - disable dropout

        self.freeze()         # default to production mode - disable training

        return self



    def print_params(self):

        print(self.__class__.__name__)

        print(self)

        for name, parameter in sorted(self.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):

            print(name)

            print(re.sub(r'\n( *\n)+', '\n', str(parameter.data.cpu().numpy())))  # remove extranious newlines

            print()







    ### Casting



    def cast_bool(self, x: torch.Tensor) -> torch.Tensor:

        # noinspection PyTypeChecker

        return (x > 0.5)



    def cast_int(self, x: torch.Tensor) -> torch.Tensor:

        return self.cast_bool(x).to(torch.int8)



    def cast_int_float(self, x: torch.Tensor) -> torch.Tensor:

        return self.cast_bool(x).to(torch.float32).requires_grad_(True)





    def cast_to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        if torch.is_tensor(x):

            return x.to(torch.float32).to(device)

        if isinstance(x, list):

            x = np.array(x)

        if isinstance(x, np.ndarray):

            x = torch.from_numpy(x).to(torch.float32)

            x = x.to(device)

            return x  # x.shape = (42,3)

        raise TypeError(f'{self.__class__.__name__}.cast_to_tensor() invalid type(x) = {type(x)}')





    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)

    # tensorflow requires: channels_last     = (batch_size, height, width, channels)

    def cast_inputs(self, x: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> torch.Tensor:

        x = self.cast_to_tensor(x)

        if x.dim() == 1:             # single row from dataframe

            x = x.view(1, 1, math.isqrt(x.shape[0]), math.isqrt(x.shape[0]))

        elif x.dim() == 2:

            if x.shape[0] == x.shape[1]:  # single 2d board

                x = x.view((1, 1, x.shape[0], x.shape[1]))

            else: # rows of flattened boards

                x = x.view((-1, 1, math.isqrt(x.shape[1]), math.isqrt(x.shape[1])))

        elif x.dim() == 3:                                        # numpy  == (batch_size, height, width)

            x = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))   # x.shape = (batch_size, channels, height, width)

        elif x.dim() == 4:

            pass  # already in (batch_size, channels, height, width) format, so do nothing

        return x

import torch.nn as nn

import torch.nn.functional as F





class ReLU1(nn.Module):

    def forward(self, x):

        return F.relu6(x * 6.0) / 6.0



class ReLUX(nn.Module):

    def __init__(self, max_value: float=1.0):

        super(ReLUX, self).__init__()

        self.max_value = float(max_value)

        self.scale     = 6.0 / self.max_value



    def forward(self, x):

        return F.relu6(x * self.scale) / self.scale
from abc import ABCMeta

from typing import TypeVar



import torch.nn as nn



# from neural_networks.GameOfLifeBase import GameOfLifeBase

# from neural_networks.modules.ReLUX import ReLU1





# noinspection PyTypeChecker

T = TypeVar('T', bound='GameOfLifeHardcoded')

class GameOfLifeHardcoded(GameOfLifeBase, metaclass=ABCMeta):

    """

    This implements the life_step() function as a minimalist Neural Network function with hardcoded weights

    Subclasses implement the effect of different activation functions and weights

    """

    def __init__(self):

        super().__init__()



        self.input      = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)

        self.counter    = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),

                                    padding=1, padding_mode='circular', bias=False)

        self.logics     = nn.ModuleList([

            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))

        ])

        self.output     = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))

        self.activation = nn.Identity()

        self.trainable_layers = [ 'input' ]  # we need at least one trainable layer

        self.criterion  = nn.MSELoss()





    def forward(self, x):

        x = input = self.cast_inputs(x)



        x = self.input(x)     # noop - a single node linear layer - torch needs at least one trainable layer

        x = self.counter(x)   # counter counts above 6, so no ReLU6



        for logic in self.logics:

            x = logic(x)

            x = self.activation(x)



        x = self.output(x)

        # x = torch.sigmoid(x)

        x = ReLU1()(x)  # we actually want a ReLU1 activation for binary outputs



        return x







    # Load / Save Functionality



    def load(self, load_weights=False):

        return super().load(load_weights=False)

    

    def load_state_dict(self, **kwargs):

        return self



    def save(self):

        return self



    def unfreeze(self: T) -> T:

        super().unfreeze()

        self.freeze()

        for trainable_layer_name in self.trainable_layers:

            for name, parameter in self.named_parameters():

                if name.startswith( trainable_layer_name ):

                    parameter.requires_grad = True

        return self
from abc import ABCMeta



import torch

import torch.nn as nn



# from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded

# from neural_networks.modules.ReLUX import ReLU1





class GameOfLifeHardcodedReLU1_21(GameOfLifeHardcoded, metaclass=ABCMeta):

    """

    This uses ReLU1 as binary true/false activation layer to implement the game of life rules using 2 nodes:

    AND(

        z3.AtLeast( past_cell, *past_neighbours, 3 ): n >= 3

        z3.AtMost(             *past_neighbours, 3 ): n <  4

    )



    This network trained from random weights:

    - is capable of: learning weights for the output layer after 400-600 epochs (occasionally much less)

    - has trouble learning the self.logic[0] AND gate weights (without lottery ticket initialization)



    This paper discusses lottery-ticket weight initialization and the issues behind auto-learning hardcoded solutions:

    - Paper: It's Hard for Neural Networks To Learn the Game of Life - https://arxiv.org/abs/2009.01398



    See GameOfLifeHardcodedReLU1_41 for an alternative implementation using 4 nodes

    """

    def __init__(self):

        super().__init__()



        self.trainable_layers  = [ 'input', 'logics', 'output' ]

        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer

        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),

                                  padding=1, padding_mode='circular', bias=False)

        self.logics  = nn.ModuleList([

            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))

        ])

        self.output  = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))

        self.activation = ReLU1()





    def load(self, load_weights=False):

        super().load(load_weights=load_weights)



        self.input.weight.data   = torch.tensor([[[[1.0]]]])

        self.counter.weight.data = torch.tensor([

            [[[ 0.0, 0.0, 0.0 ],

              [ 0.0, 1.0, 0.0 ],

              [ 0.0, 0.0, 0.0 ]]],



            [[[ 1.0, 1.0, 1.0 ],

              [ 1.0, 0.0, 1.0 ],

              [ 1.0, 1.0, 1.0 ]]]

        ])



        self.logics[0].weight.data = torch.tensor([

            [[[ 1.0 ]], [[  1.0 ]]],   # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),

            [[[ 0.0 ]], [[ -1.0 ]]],   # n <  4   # z3.AtMost(             *past_neighbours, 3 ),

        ])

        self.logics[0].bias.data = torch.tensor([

            -3.0 + 1.0,               # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),

            +3.0 + 1.0,               # n <= 3   # z3.AtMost(             *past_neighbours, 3 ),

        ])



        # Both of the statements need to be true, and ReLU1 enforces we can't go above 1

        self.output.weight.data = torch.tensor([[

            [[  1.0 ]],

            [[  1.0 ]],

        ]])

        self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2



        self.to(self.device)

        return self





    ### kaiming corrects for mean and std of the ReLU function (V shaped), but we are using ReLU1 (Z shaped)

    ### normal distribution seems to perform slightly better than uniform

    ### default initialization actually seems to perform better than both kaiming and xavier

    # nn.init.xavier_uniform_(layer.weight)   # 600, 141, 577, 581, 583, epochs for output to train

    # nn.init.kaiming_uniform_(layer.weight)  # 664, 559, 570, 592, 533

    # nn.init.kaiming_normal_(layer.weight)   # 450, 562, 456, 164, 557

    # nn.init.xavier_normal_(layer.weight)    # 497, 492, 583, 461, 475

    # default (pass) initialization:          # 232, 488,  43, 333,  412

    def weights_init(self, layer):

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):

            pass  # use default initialization







if __name__ == '__main__':

    import numpy as np



    model = GameOfLifeHardcodedReLU1_21()

    model.print_params()



    board = np.array([

        [0,0,0,0,0],

        [0,0,0,0,0],

        [0,1,1,1,0],

        [0,0,0,0,0],

        [0,0,0,0,0],

    ])

    result1 = model.predict(board)

    result2 = model.predict(result1)

    assert np.array_equal(board, result2)

    print('Test passed!')
from typing import TypeVar



import torch

import torch.nn as nn

from torch import tensor



# from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded



# noinspection PyTypeChecker

T = TypeVar('T', bound='GameOfLifeHardcodedTanh')

class GameOfLifeHardcodedTanh(GameOfLifeHardcoded):

    """

    This uses Tanh as binary true/false activation layer to implement the game of life rules using 4 nodes:

    AND(

        z3.AtLeast( past_cell, *past_neighbours, 3 ): n >= 3

        z3.AtMost(             *past_neighbours, 3 ): n <  4

    )



    Tanh() is both applied to input data as well as being the activation function for the output

    ReLU1  is still being used as the activation function for the logic_games layers



    In theory, the idea was that this would hopefully make this implementation more robust when dealing with

    non-saturated inputs (ie 0.8 rather than 1.0).

    A continual smooth gradient may (hopefully) assist OuroborosLife in using this both as a loss function

    and as a hidden layer inside its own CNN layers. I suspect a ReLU gradient of 0 may be causing problems.



    In practice, the trained tanh solution converged to using two different order of magnitude scales,

    similar to the manual implementation GameOfLifeHardcodedReLU1_41.py.



    I am unsure if this is make the algorithm more or less stable to non-saturated 0.8 inputs.

    However the final tanh() will produce mostly saturated outputs.



    Trained solution

        input.weight      1.0

        logics.0.weight  [[  2.163727,  2.2645657  ]

                          [ -0.018100, -0.29718676 ]]

        logics.0.bias     [ -2.189014,  2.1635942  ]

        output.weight     [  8.673016,  9.106407   ]

        output.bias        -15.924878,





    See GameOfLifeHardcodedReLU1_21 for an alternative implementation using only 2 nodes

    """

    def __init__(self):

        super().__init__()



        self.trainable_layers  = [ 'logics', 'outputs' ]

        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),

                                  padding=1, padding_mode='circular', bias=False)

        self.logics  = nn.ModuleList([

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1,1))

        ])

        self.output  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))





    def forward(self, x):

        x = input = self.cast_inputs(x)



        x = torch.tanh(x)    # tanh

        x = self.input(x)    # noop - a single node linear layer - torch needs at least one trainable layer

        x = self.counter(x)  # counter counts above 6, so no ReLU6



        for logic in self.logics:

            x = logic(x)

            x = torch.tanh(x)

            # x = ReLU1()(x)  # ReLU1 is needed to make the logic gates work



        x = self.output(x)

        x = torch.tanh(x)



        return x





    def load(self: T) -> T:

        super().load()

        # self.input.weight.data   = tensor([[[[ 1.0498226 ]]]])

        self.input.weight.data   = tensor([[[[ 1.0 ]]]])

        self.counter.weight.data = tensor([

            [[[ 0.0, 0.0, 0.0 ],

              [ 0.0, 1.0, 0.0 ],

              [ 0.0, 0.0, 0.0 ]]],



            [[[ 1.0, 1.0, 1.0 ],

              [ 1.0, 0.0, 1.0 ],

              [ 1.0, 1.0, 1.0 ]]]

        ]) / self.activation(tensor([ 1.0 ]))



        self.logics[0].weight.data = tensor([

            [[[  2.077 ]], [[  2.197 ]]],   # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),

            [[[ -0.020 ]], [[ -0.250  ]]],  # n <  4   # z3.AtMost(             *past_neighbours, 3 ),

        ])

        self.logics[0].bias.data = tensor([

            -2.022,              # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),

             1.978,              # n <= 3   # z3.AtMost(             *past_neighbours, 3 ),

        ])



        # Both of the statements need to be true. Tanh after logics has the domain (-1,1)

        # Weights here also need to be sufficiently large to saturate sigmoid()

        self.output.weight.data = tensor([[

            [[ 9.0 ]],

            [[ 9.0 ]],

        ]])

        self.output.bias.data = tensor([ -16.0 ])  # sum() > 1.5 as tanh()'ed inputs may not be at full saturation



        self.to(self.device)

        return self





if __name__ == '__main__':

    import numpy as np



    model = GameOfLifeHardcodedTanh()

    model.print_params()



    board = np.array([

        [0,0,0,0,0],

        [0,0,0,0,0],

        [0,1,1,1,0],

        [0,0,0,0,0],

        [0,0,0,0,0],

    ])

    result1 = model.predict(board)

    result2 = model.predict(result1)

    assert np.array_equal(board, result2)

    print('Test passed!')

import torch.nn as nn

import torch.nn.functional as F





class ReLU1(nn.Module):

    def forward(self, x):

        return F.relu6(x * 6.0) / 6.0



class ReLUX(nn.Module):

    def __init__(self, max_value: float=1.0):

        super(ReLUX, self).__init__()

        self.max_value = float(max_value)

        self.scale     = 6.0 / self.max_value



    def forward(self, x):

        return F.relu6(x * self.scale) / self.scale
from abc import ABCMeta



import torch

import torch.nn as nn



# from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded

# from neural_networks.modules.ReLUX import ReLU1





class GameOfLifeHardcodedReLU1_41(GameOfLifeHardcoded, metaclass=ABCMeta):

    """

    This uses ReLU1 as binary true/false activation layer to implement the game of life rules using 4 nodes:

    SUM(

        Alive && neighbours >= 2

        Alive && neighbours <= 3

        Dead  && neighbours >= 3

        Dead  && neighbours <= 3

    ) >= 2



    Alive! is implemented as -10 weight, which is greater than maximum value of the 3x3-1=8 counter convolution

    sum() >= 2 works because Dead and Alive are mutually exclusive conditions



    See GameOfLifeHardcodedReLU1_21 for an alternative implementation using only 2 nodes

    """



    def __init__(self):

        super().__init__()



        self.trainable_layers  = [ 'input', 'output' ]

        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer

        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),

                                  padding=1, padding_mode='circular', bias=False)

        self.logics  = nn.ModuleList([

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1,1))

        ])

        self.output  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))

        self.activation = ReLU1()







    def load(self, load_weights=False):

        super().load(load_weights=load_weights)



        self.input.weight.data   = torch.tensor([[[[1.0]]]])

        self.counter.weight.data = torch.tensor([

            [[[ 0.0, 0.0, 0.0 ],

              [ 0.0, 1.0, 0.0 ],

              [ 0.0, 0.0, 0.0 ]]],



            [[[ 1.0, 1.0, 1.0 ],

              [ 1.0, 0.0, 1.0 ],

              [ 1.0, 1.0, 1.0 ]]]

        ])



        self.logics[0].weight.data = torch.tensor([

            [[[  10.0 ]], [[  1.0 ]]],  # Alive && neighbours >= 2

            [[[  10.0 ]], [[ -1.0 ]]],  # Alive && neighbours <= 3

            [[[ -10.0 ]], [[  1.0 ]]],  # Dead  && neighbours >= 3

            [[[ -10.0 ]], [[ -1.0 ]]],  # Dead  && neighbours <= 3

        ])

        self.logics[0].bias.data = torch.tensor([

            -10.0 - 2.0 + 1.0,  # Alive +  neighbours >= 2

            -10.0 + 3.0 + 1.0,  # Alive + -neighbours <= 3

              0.0 - 3.0 + 1.0,  # Dead  +  neighbours >= 3

              0.0 + 3.0 + 1.0,  # Dead  + -neighbours <= 3

        ])



        # Both of the Alive or Dead statements need to be true

        #   sum() >= 2 works here because the -10 weight above makes the two clauses mutually exclusive

        # Otherwise it would require a second layer to formally implement:

        #   OR( AND(input[0], input[1]), AND(input[3], input[4]) )

        self.output.weight.data = torch.tensor([[

            [[  1.0 ]],

            [[  1.0 ]],

            [[  1.0 ]],

            [[  1.0 ]],

        ]])

        self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2



        self.to(self.device)

        return self







if __name__ == '__main__':

    import numpy as np



    model = GameOfLifeHardcodedReLU1_41()

    model.print_params()



    board = np.array([

        [0,0,0,0,0],

        [0,0,0,0,0],

        [0,1,1,1,0],

        [0,0,0,0,0],

        [0,0,0,0,0],

    ])

    result1 = model.predict(board)

    result2 = model.predict(result1)

    assert np.array_equal(board, result2)

    print('Test passed!')
# Functions for implementing Game of Life Forward Play

from typing import List



import numpy as np

import scipy.sparse

from joblib import delayed

from joblib import Parallel

from numba import njit





# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_numpy(X: np.ndarray):

    """Game of life step using generator expressions"""

    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)

                     for i in (-1, 0, 1) for j in (-1, 0, 1)

                     if (i != 0 or j != 0))

    return (nbrs_count == 3) | (X & (nbrs_count == 2))





# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_scipy(X: np.ndarray):

    """Game of life step using scipy tools"""

    from scipy.signal import convolve2d

    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X

    return (nbrs_count == 3) | (X & (nbrs_count == 2))







# NOTE: @njit doesn't like np.roll(axis=) so reimplement explictly

@njit

def life_neighbours_xy(board: np.ndarray, x, y, max_value=3):

    size_x = board.shape[0]

    size_y = board.shape[1]

    neighbours = 0

    for i in (-1, 0, 1):

        for j in (-1, 0, 1):

            if i == j == 0: continue    # ignore self

            xi = (x + i) % size_x

            yj = (y + j) % size_y

            neighbours += board[xi, yj]

            if neighbours > max_value:  # shortcircuit return 4 if overpopulated

                return neighbours

    return neighbours





@njit

def life_neighbours(board: np.ndarray, max_value=3):

    size_x = board.shape[0]

    size_y = board.shape[1]

    output = np.zeros(board.shape, dtype=np.int8)

    for x in range(size_x):

        for y in range(size_y):

            output[x,y] = life_neighbours_xy(board, x, y, max_value)

    return output





@njit

def life_step_njit(board: np.ndarray) -> np.ndarray:

    """Game of life step using generator expressions"""

    size_x = board.shape[0]

    size_y = board.shape[1]

    output = np.zeros(board.shape, dtype=np.int8)

    for x in range(size_x):

        for y in range(size_y):

            cell       = board[x,y]

            neighbours = life_neighbours_xy(board, x, y, max_value=3)

            if ( (cell == 0 and      neighbours == 3 )

              or (cell == 1 and 2 <= neighbours <= 3 )

            ):

                output[x, y] = 1

    return output



life_step = life_step_njit  # create global alias

def life_steps(boards: List[np.ndarray]) -> List[np.ndarray]:

    """ Parallel version of life_step() but for an array of boards """

    return Parallel(-1)( delayed(life_step)(board) for board in boards )





@njit

def life_step_delta(board: np.ndarray, delta):

    for t in range(delta): board = life_step(board)

    return board





def life_step_3d(board: np.ndarray, delta):

    solution_3d = np.array([ board ], dtype=np.int8)

    for t in range(delta):

        board       = life_step(board)

        solution_3d = np.append( solution_3d, [ board ], axis=0)

    return solution_3d





# RULES: https://www.kaggle.com/c/conway-s-reverse-game-of-life/data

def generate_random_board(shape=(25,25)):

    # An initial board was chosen by filling the board with a random density between 1% full (mostly zeros) and 99% full (mostly ones).

    # DOCS: https://cmdlinetips.com/2019/02/how-to-create-random-sparse-matrix-of-specific-density/

    density = np.random.random() * 0.98 + 0.01

    board   = scipy.sparse.random(*shape, density=density, data_rvs=np.ones).toarray().astype(np.int8)



    # The starting board's state was recorded after the 5 "warmup steps". These are the values in the start variables.

    for t in range(5):

        board = life_step(board)

        if np.count_nonzero(board) == 0:

            return generate_random_board(shape)  # exclude empty boards and try again

    return board



def generate_random_boards(count, shape=(25,25)):

    generated_boards = Parallel(-1)( delayed(generate_random_board)(shape) for _ in range(count) )

    return generated_boards

import pytest 



models = [

    GameOfLifeHardcodedReLU1_41(),

    GameOfLifeHardcodedReLU1_21(),

    GameOfLifeHardcodedTanh(),

]



# @pytest.mark.parametrize("model", models)

def test_GameOfLifeHardcoded_generated_boards(model):

    inputs   = generate_random_boards(100_000)

    expected = life_steps(inputs)

    outputs  = model.predict(inputs)

    assert np.array_equal( outputs, expected )  # assert 100% accuracy



for model in models:

    test_GameOfLifeHardcoded_generated_boards

    print(f'{model.__class__.__name__:27s} - 100,000 tests passed')
def profile_GameOfLifeHardcoded():

    import timeit

    import operator

    # from utils.game import generate_random_boards, life_step, life_step_1, life_step_2



    gameOfLifeHardcodedReLU1_41 = GameOfLifeHardcodedReLU1_41().load().to(device)

    gameOfLifeHardcodedReLU1_21 = GameOfLifeHardcodedReLU1_21().load().to(device)

    gameOfLifeHardcodedTanh     = GameOfLifeHardcodedTanh().load().to(device)

    

    for batch_size in [1, 1_000]:

        boards  = generate_random_boards(batch_size)

        number  = 10

        timings = {

            'GameOfLifeHardcodedReLU1_41() - batch': timeit.timeit(lambda:   gameOfLifeHardcodedReLU1_41.predict(boards),                      number=number),

            'GameOfLifeHardcodedReLU1_21() - batch': timeit.timeit(lambda:   gameOfLifeHardcodedReLU1_41.predict(boards),                      number=number),

            'GameOfLifeHardcodedTanh()     - batch': timeit.timeit(lambda:   gameOfLifeHardcodedTanh.predict(boards),                          number=number),

            'GameOfLifeHardcodedReLU1_41() - loop':  timeit.timeit(lambda: [ gameOfLifeHardcodedReLU1_21.predict(board) for board in boards ], number=number),

            'GameOfLifeHardcodedReLU1_21() - loop':  timeit.timeit(lambda: [ gameOfLifeHardcodedReLU1_21.predict(board) for board in boards ], number=number),

            'GameOfLifeHardcodedTanh()     - loop':  timeit.timeit(lambda: [ gameOfLifeHardcodedTanh.predict(board)     for board in boards ], number=number),

            'life_step() - njit':                    timeit.timeit(lambda: [ life_step_njit(board)                      for board in boards ], number=number),

            'life_step() - numpy':                   timeit.timeit(lambda: [ life_step_numpy(board)                     for board in boards ], number=number),

            'life_step() - scipy':                   timeit.timeit(lambda: [ life_step_scipy(board)                     for board in boards ], number=number),

        }

        print(f'{device} | batch_size = {len(boards)}')

        for key, value in sorted(timings.items(), key=operator.itemgetter(1)):

            print(f'- {key:37s} = {value/number/len(boards) * 1_000:6.3f}ms')

        print()

        

        

device = torch.device("cpu")

profile_GameOfLifeHardcoded()



if torch.cuda.is_available():

    device = torch.device("cuda:0")

    profile_GameOfLifeHardcoded()
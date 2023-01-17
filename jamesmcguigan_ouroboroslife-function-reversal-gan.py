# Functions for implementing Game of Life Forward Play

from typing import List



import numpy as np

import scipy.sparse

from joblib import delayed

from joblib import Parallel

from numba import njit





# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_1(X: np.ndarray):

    """Game of life step using generator expressions"""

    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)

                     for i in (-1, 0, 1) for j in (-1, 0, 1)

                     if (i != 0 or j != 0))

    return (nbrs_count == 3) | (X & (nbrs_count == 2))





# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_2(X: np.ndarray):

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

def life_step(board: np.ndarray) -> np.ndarray:

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

# Modified from Source: https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

# Switched from using F.cross_entropy() to F.binary_cross_entropy()

import torch



import torch.nn as nn





class FocalLoss(nn.Module):



    def __init__(self, focusing_param=2, balance_param=0.25):

        super(FocalLoss, self).__init__()



        self.focusing_param = focusing_param

        self.balance_param  = balance_param

        self.bce            = nn.BCELoss()



    def forward(self, output, target):

        logpt      = - self.bce(output, target)

        pt         = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss
device   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

__file__ = './notebook.py'
from __future__ import annotations



import os

from abc import ABCMeta

from typing import List

from typing import TypeVar

from typing import Union



import humanize

import numpy as np

import torch

import torch.nn as nn



# from neural_networks.device import device



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





    @staticmethod

    def weights_init(layer):

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):

            nn.init.kaiming_normal_(layer.weight)

            nn.init.constant_(layer.bias, 0.1)



    ### Prediction



    def __call__(self, *args, **kwargs) -> torch.Tensor:

        if not self.loaded: self.load()  # autoload on first function call

        return super().__call__(*args, **kwargs)



    def predict(self, inputs: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> np.ndarray:

        """ Wrapper function around __call__() that returns a numpy int8 array for external usage """

        outputs = self(inputs)

        outputs = self.cast_int(outputs).squeeze().cpu().numpy()

        return outputs







    ### Training



    def loss(self, outputs, expected, input):

        return self.criterion(outputs, expected)



    def accuracy(self, outputs, expected, inputs) -> float:

        # noinspection PyTypeChecker

        return torch.sum(self.cast_int(outputs) == self.cast_int(expected)).cpu().numpy() / np.prod(outputs.shape)







    ### Freee / Unfreeze



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





    def load(self: T) -> T:

        if os.path.exists(self.filename):

            try:

                self.load_state_dict(torch.load(self.filename))

                print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')

            except Exception as exception:

                # Ignore errors caused by model size mismatch

                print(f'{self.__class__.__name__}.load(): model has changed dimensions, reinitializing weights\n')

                self.apply(self.weights_init)

        else:

            print(f'{self.__class__.__name__}.load(): model file not found, reinitializing weights\n')

            self.apply(self.weights_init)



        self.loaded = True    # prevent any infinite if self.loaded loops

        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU

        self.eval()           # default to production mode - disable dropout

        self.freeze()         # default to production mode - disable training

        return self







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

            x = x.view(1, 1, torch.sqrt(x.shape[0]), torch.sqrt(x.shape[0]))

        elif x.dim() == 2:

            if x.shape[0] == x.shape[1]:  # single 2d board

                x = x.view(1, 1, x.shape[0], x.shape[1])

            else: # rows of flattened boards

                x = x.view(-1, 1, torch.sqrt(x.shape[1]), torch.sqrt(x.shape[1]))

        elif x.dim() == 3:                                 # numpy  == (batch_size, height, width)

            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])   # x.shape = (batch_size, channels, height, width)

        elif x.dim() == 4:

            pass  # already in (batch_size, channels, height, width) format, so do nothing

        return x
import atexit

import gc

import time

from typing import List

from typing import Tuple

from typing import Union



import numpy as np

import torch

import torch as pt

import torch.nn as nn



# from neural_networks.FocalLoss import FocalLoss

# from neural_networks.GameOfLifeBase import GameOfLifeBase

# from utils.game import generate_random_board

# from utils.game import life_step_3d





class OuroborosLife(GameOfLifeBase):

    """

    The idea of the Ouroboros Network is that rather than just predicting the next or previous state,

    we want to past, present and future simultaneously in the same network.



    The dataset is a sequence of 3 consecutive board states generated by life_step().



    The network takes the middle/present board state and attempts to predict all Past, Present and Future states



    The loss function computes the loss against the original training data, but also feeds back in upon itself.

    The output for Future is fed back in and it's Past is compared with the Present, likewise in reverse with the Paspt.

    """

    @property

    def filename(self) -> str:

        """ ./models/OuroborosLife3.pth || ./models/OuroborosLife5.pth """

        return super().filename.replace('.pth', f'{self.out_channels}.pth')





    def __init__(self, in_channels=1, out_channels=3):

        """

        TODO:

        - Create split blocks that return: [identity, avgpool, maxpool ]  # do we need 3x3 convolution with fixed weights

        - Basically find a way to count neighbours

        - reduce model size

        - add in dense layer at end + middle (as opposed to deconvolution???)

        """

        assert out_channels % 2 == 1, f'{self.__class__.__name__}(out_channels={out_channels}) must be odd'



        super().__init__()

        self.in_channels  = in_channels

        self.out_channels = out_channels  # Past, Present and Future



        self.relu    = nn.LeakyReLU()     # combines with nn.init.kaiming_normal_()

        self.dropout = nn.Dropout(p=0.2)



        # 2**9 = 512 filters and kernel size of 3x3 to allow for full encoding of game rules

        # Pixels can see distance 5 neighbours, (hopefully) sufficient for delta=2 timesteps or out_channels=5

        # https://www.youtube.com/watch?v=H3g26EVADgY&feature=youtu.be&t=1h39m410s&ab_channel=JeremyHoward

        self.cnn_layers = nn.ModuleList([

            # Previous pixel state requires information from distance 2, so we need two 3x3 convolutions

            nn.Conv2d(in_channels=in_channels, out_channels=512,  kernel_size=(5,5), padding=2, padding_mode='circular'),

            nn.Conv2d(in_channels=512,   out_channels=256,  kernel_size=(1,1)),

            nn.Conv2d(in_channels=256,   out_channels=128,  kernel_size=(1,1)),



            nn.Conv2d(in_channels=1+128, out_channels=128,  kernel_size=(3,3), padding=1, padding_mode='circular'),

            nn.Conv2d(in_channels=128,   out_channels=512,  kernel_size=(1,1)),

            nn.Conv2d(in_channels=512,   out_channels=256,  kernel_size=(1,1)),

            nn.Conv2d(in_channels=256,   out_channels=128,  kernel_size=(1,1)),



            # # Deconvolution + Convolution allows neighbouring pixels to share information to simulate forward play

            # # This creates a 52x52 grid of interspersed cells that can then be downsampled back down to 25x25

            nn.ConvTranspose2d(in_channels=1+128, out_channels=512,  kernel_size=(3,3), stride=2, dilation=1),

            nn.Conv2d(in_channels=512,   out_channels=256,   kernel_size=(1,1)),

            nn.Conv2d(in_channels=256,   out_channels=64,    kernel_size=(1,1)),

            nn.Conv2d(in_channels=64,    out_channels=128,   kernel_size=(3,3), stride=2),  # undo deconvolution



            nn.Conv2d(in_channels=1+128, out_channels=64,    kernel_size=(1,1)),

            nn.Conv2d(in_channels=64,    out_channels=32,    kernel_size=(1,1)),

            nn.Conv2d(in_channels=32,    out_channels=16,    kernel_size=(1,1)),

            nn.Conv2d(in_channels=1+16,  out_channels=out_channels, kernel_size=(1,1)),

        ])

        self.batchnorm_layers = nn.ModuleList([

            nn.BatchNorm2d(cnn_layer.out_channels)

            for cnn_layer in self.cnn_layers

        ])





        # self.criterion = nn.BCELoss()

        self.criterion = FocalLoss()

        # self.criterion = nn.MSELoss()

        self.optimizer = pt.optim.RMSprop(self.parameters(), lr=0.01, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(

            self.optimizer,

            max_lr=1e-3,

            base_lr=1e-5,

            step_size_up=10,

            mode='exp_range',

            gamma=0.8

        )



    # def load(self):

    #     super().load()

    #     self.apply(self.weights_init)





    def forward(self, x):

        x = input = self.cast_inputs(x)

        for n, (cnn_layer, batchnorm_layer) in enumerate(zip(self.cnn_layers, self.batchnorm_layers)):

            if cnn_layer.in_channels > 1 and cnn_layer.in_channels % 2 == 1:   # autodetect 1+in_channels == odd number

                x = torch.cat([ x, input ], dim=1)                     # passthrough original cell state

            x = cnn_layer(x)

            if n != len(self.cnn_layers)-1:

                x = self.relu(x)

                if n != 1:               # Don't apply dropout to the first layer

                    x = self.dropout(x)  # BatchNorm eliminates the need for Dropout in some cases cause BN provides similar regularization benefits as Dropout intuitively"

                x = batchnorm_layer(x)   # batchnorm goes after activation

            else:

                x = torch.sigmoid(x)  # output requires sigmoid activation

        return x





    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)

    # tensorflow requires: channels_last     = (batch_size, height, width, channels)

    def cast_inputs(self, x: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> torch.Tensor:

        x = self.cast_to_tensor(x)

        if   x.dim() == 4: pass

        elif x.dim() == 3 and x.shape[0] == self.out_channels:           # x.shape = (channels, height, width)

            x = x.view(1, self.in_channels, x.shape[1], x.shape[2])   # x.shape = (batch_size, channels, height, width)

        else:

            x = super().cast_inputs(x)

        return x





    def loss_dataset(self, outputs, timeline, inputs, exclude_past=True):

        # Exclude past losses to avoid the many-pasts to one-future problem

        if exclude_past:

            t_present = self.out_channels//2

            outputs   = outputs[  :, t_present:, :, : ]

            timeline  = timeline[ :, t_present:, :, : ]



        ### Other ways of computing dataset loss

        # dataset_loss = torch.mean(torch.mean(( (timeline-outputs)**2 ).flatten(1), dim=1)) # average MSE per timeframe

        # dataset_loss = torch.sum(torch.tensor([

        #     self.criterion(outputs[b][t], timeline[b][t])  # NOTE: FocalLoss(outputs, target) needed in correct order

        #     for b in range(timeline.shape[0])

        #     for t in range(timeline.shape[1])

        # ], requires_grad=True))



        dataset_loss = self.criterion(outputs, timeline)

        return dataset_loss





    def loss_accuracy_ouroboros(self, outputs, timeline, inputs) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:

        """

        Compute simplified losses for each head, only comparing reoutputs with timeline[t_present]

        reinput    = t1=0 = Past | t1=1 = Present | t1=2 = Future

        reoutput   = [ Past2, Past, Present ]@t=0, [ Past, Present, Future ]@t=1, [ Present, Future, Future2 ]@t=2

        """

        losses     = pt.zeros(self.out_channels, dtype=pt.float32, requires_grad=True).to(self.device)

        acc_boards = pt.zeros(self.out_channels, dtype=pt.float32, requires_grad=False).to(self.device)

        acc_pixels = pt.zeros(self.out_channels, dtype=pt.float32, requires_grad=False).to(self.device)

        for t_input in range(self.out_channels):

            t_present = self.out_channels//2 - t_input

            reinput   = outputs[:,t_input,:,:].unsqueeze(1)

            reoutputs = self(reinput)



            # Losses get calculated for present and all future datapoints

            for d in range(self.out_channels):

                if reoutputs.shape[1] <= t_present + d: break

                if timeline.shape[1]  <= t_input   + d: break

                losses[t_input] += self.criterion(reoutputs[:,t_present+d,:,:], timeline[:,t_input+d,:,:])



            # Accuracy is based only on the self-referential present

            pixels_correct       = ((reoutputs[:,t_present,:,:] > 0.5) == (timeline[:,t_input,:,:] > 0.5)).to(pt.float).detach()

            acc_pixels[t_input] +=  pixels_correct.mean()

            acc_boards[t_input] += (pixels_correct.mean(dim=1) == 1.0).to(pt.float).mean()

        return losses, acc_pixels, acc_boards









    def fit(self, epochs=100_000, batch_size=25, max_delta=25, timeout=0):

        gc.collect()

        torch.cuda.empty_cache()

        atexit.register(model.save)

        self.train()

        self.unfreeze()

        print(self)

        try:

            # timelines_batch = np.array([

            #     life_step_3d(generate_random_board(), max_delta)

            #     for _ in range(batch_size)

            # ])

            time_start  = time.perf_counter()

            board_count = 0

            dataset_accuracies = [0]

            for epoch in range(1, epochs+1):

                if np.min(dataset_accuracies[-10:]) == 1.0: break  # we have reached 100% accuracy

                if timeout and timeout < time.perf_counter() - time_start: break



                epoch_start = time.perf_counter()

                timelines_batch = np.array([

                    life_step_3d(generate_random_board(), max_delta)

                    for _ in range(batch_size)

                ])

                epoch_ds_losses  = []

                epoch_losses     = []

                epoch_acc_pixels = []

                epoch_acc_boards = []

                d = self.out_channels // 2  # In theory this should work for 5 or 7 channels

                for t in range(d, max_delta - d):

                    inputs_np   = timelines_batch[:, np.newaxis, t,:,:]  # (batch_size=10, channels=1,  width=25, height=25)

                    timeline_np = timelines_batch[:, t-d:t+d+1,    :,:]  # (batch_size=10, channels=10, width=25, height=25)

                    inputs      = pt.tensor(inputs_np).to(self.device).to(pt.float32)

                    timeline    = pt.tensor(timeline_np).to(self.device).to(pt.float32)



                    self.optimizer.zero_grad()

                    outputs      = self(inputs)

                    dataset_loss = self.loss_dataset(outputs, timeline, inputs, exclude_past=True)

                    orb_losses, acc_pixels, acc_boards = model.loss_accuracy_ouroboros(outputs, timeline, inputs)

                    loss = pt.mean(orb_losses) + dataset_loss

                    loss.backward()

                    self.optimizer.step()

                    self.scheduler.step()



                    board_count += batch_size

                    epoch_ds_losses.append(dataset_loss.detach().item())

                    epoch_losses.append(orb_losses.detach())

                    epoch_acc_pixels.append(acc_pixels.detach())

                    epoch_acc_boards.append(acc_boards.detach())

                    torch.cuda.empty_cache()



                dataset_accuracies.append( pt.stack(epoch_acc_boards).min() )

                epoch_loss      = f"{100*np.mean(epoch_ds_losses):.6f} : " + " ".join([ f'{100*n:.6f}' for n in pt.stack(epoch_losses).mean(dim=0).tolist()     ])

                epoch_acc_pixel = " ".join([ f'{n:.3f}'     for n in pt.stack(epoch_acc_pixels).mean(dim=0).tolist() ])

                epoch_acc_board = " ".join([ f'{n:.3f}'     for n in pt.stack(epoch_acc_boards).mean(dim=0).tolist() ])



                epoch_time = time.perf_counter() - epoch_start

                time_taken = time.perf_counter() - time_start

                if epoch <= 10 or epoch <= 100 and epoch % 10 == 0 or epoch % 100 == 0:  

                    print(f'epoch: {epoch:4d} | boards: {board_count:5d} | loss: {epoch_loss} | pixels = {epoch_acc_pixel} | boards = {epoch_acc_board} | time: {time_taken//60:.0f}:{time_taken%60:02.0f} @ {1000*epoch_time//batch_size:3.0f}ms')

                    # print(f'epoch: {epoch:4d} | boards: {board_count:5d} | loss: {np.mean(epoch_losses):.6f} | ouroboros: {np.mean(ouroboros_losses):.6f} | dataset: {np.mean(dataset_losses):.6f} | accuracy = {np.mean(epoch_accuracies):.6f} | time: {1000*epoch_time//batch_size}ms/board | {time_taken//60:.0f}:{time_taken%60:02.0f}')

        except KeyboardInterrupt: pass

        finally:

            model.save()

            atexit.unregister(model.save)

            torch.cuda.empty_cache()

            gc.collect()

# !cp -rv ../input/ouroboroslife-function-reversal-gan/models ./

# !cp -v  ../input/ouroboroslife-function-reversal-gan/*.csv  ./
model = OuroborosLife()

model.fit(timeout=1.5*60*60)
from typing import Dict



import numpy as np

import pandas as pd

from fastcache import clru_cache





@clru_cache(None)

def csv_column_names(key='start'):

    return [ f'{key}_{n}' for n in range(25**2) ]





def csv_to_delta(df, idx):

    return int(df.loc[idx]['delta'])



def csv_to_delta_list(df):

    return df['delta'].values





def csv_to_numpy(df, idx, key='start') -> np.ndarray:

    try:

        columns = csv_column_names(key)

        board   = df.loc[idx][columns].values

    except:

        board = np.zeros((25, 25))

    board = board.reshape((25,25)).astype(np.int8)

    return board





def csv_to_numpy_list(df, key='start') -> np.ndarray:

    try:

        columns = csv_column_names(key)

        output  = df[columns].values.reshape(-1,25,25)

    except:

        output  = np.zeros((0,25,25))

    return output





# noinspection PyTypeChecker,PyUnresolvedReferences

def numpy_to_dict(board: np.ndarray, key='start') -> Dict:

    assert len(board.shape) == 2  # we want 2D solutions_3d[0] not 3D solutions_3d

    assert key in { 'start', 'stop' }



    board  = np.array(board).flatten().tolist()

    output = { f"{key}_{n}": board[n] for n in range(len(board))}

    return output





def numpy_to_series(board: np.ndarray, key='start') -> pd.Series:

    solution_dict = numpy_to_dict(board, key)

    return pd.Series(solution_dict)





# Source: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

def batch(iterable, n=1):

    l = len(iterable)

    for ndx in range(0, l, n):

        yield iterable[ndx:min(ndx + n, l)]
from numba import njit



@njit

def is_valid_solution(start: np.ndarray, stop: np.ndarray, delta: int) -> bool:

    # we are rewriting data, so lets double check our work

    test_board = start

    is_valid   = np.count_nonzero(test_board) != 0

    if not is_valid: return False

    for t in range(delta):

        test_board = life_step(test_board)

        is_valid   = is_valid and np.count_nonzero(test_board) != 0

        if not is_valid: return False

    is_valid = is_valid and np.all(test_board == stop)

    return is_valid



@njit

def is_valid_solution_3d(solution_3d: np.ndarray) -> bool:

    return is_valid_solution(solution_3d[0], solution_3d[-1], delta=len(solution_3d)-1)

import pandas as pd



input_directory  = '../input/conways-reverse-game-of-life-2020/'

output_directory = './'



train_file             = f'{input_directory}/train.csv'

test_file              = f'{input_directory}/test.csv'

sample_submission_file = f'{input_directory}/sample_submission.csv'

submission_file        = f'{output_directory}/submission.csv'

exact_submission_file  = f'{output_directory}/submission_exact.csv'

timeout_file           = f'{output_directory}/timeouts.csv'



train_df              = pd.read_csv(train_file,              index_col='id')

test_df               = pd.read_csv(test_file,               index_col='id')

sample_submission_df  = pd.read_csv(sample_submission_file,  index_col='id')



# try:

#     submission_df     = pd.read_csv(submission_file,         index_col='id')

# except:

#     submission_df     = sample_submission_df.copy()



# try:

#     exact_submission  = pd.read_csv(exact_submission_file,   index_col='id')

# except:

#     exact_submission  = sample_submission_df.copy()
# import time



# from constraint_satisfaction.fix_submission import is_valid_solution

# from neural_networks.OuroborosLife import OuroborosLife

# from utils.datasets import *

# from utils.util import batch

# from utils.util import csv_to_delta_list

# from utils.util import csv_to_numpy_list

# from utils.util import numpy_to_series

# import numpy as np



def ouroborors_dataframe(df: pd.DataFrame):

    time_start    = time.perf_counter()

    

    model         = OuroborosLife()

    model.load().train().unfreeze()

    submission_df     = sample_submission_df.copy()

    exact_submission  = sample_submission_df.copy()

    

    stats = {

        "boards":  { "solved": 0, "total": 0 },

        "delta":   { "solved": 0, "total": 0 },

        "dpixels": { "solved": 0, "total": 0 },

        "pixels":  { "solved": 0, "total": 0 },

    }

    for delta in range(1,5+1):

        df_delta = df[ df.delta == delta ]

        idxs     = df_delta.index

        boards   = csv_to_numpy_list(df_delta, key='stop')

        for idxs, inputs in zip(batch(idxs, 100), batch(boards, 100)):

            outputs = inputs

            for t in range(delta):

                outputs = model.predict(outputs)[:,0,:,:]

            for idx, output_board, input_board in zip(idxs, outputs, inputs):

                stats['boards']['total']   += 1

                stats['delta']['total']    += 1

                stats['pixels']['total']   += outputs.size

                stats['pixels']['solved']  += np.count_nonzero( outputs == inputs )

                stats['dpixels']['total']  += outputs.size

                stats['dpixels']['solved'] += np.count_nonzero( outputs == inputs )

                if is_valid_solution(output_board, input_board, delta):

                    stats['boards']['solved'] += 1

                    stats['delta']['solved']  += 1

                    exact_submission.loc[idx] = numpy_to_series(output_board, key='start')

                submission_df.loc[idx]        = numpy_to_series(output_board, key='start')

        time_taken = time.perf_counter() - time_start

        print(f"delta = {delta} | solved {stats['delta']['solved']:4d}/{stats['delta']['total']:5d} = {100*stats['delta']['solved']/stats['delta']['total']:4.1f}% | {100*stats['dpixels']['solved']/stats['dpixels']['total']:4.1f}% pixels | in {time_taken//60:.0f}:{time_taken%60:02.0f}")

        stats['delta']   = { "solved": 0, "total": 0 }

        stats['dpixels'] = { "solved": 0, "total": 0 }



    time_taken = time.perf_counter() - time_start

    print(f"ouroborors_dataframe() - solved {stats['boards']['solved']:4d}/{stats['boards']['total']:5d} = {100*stats['boards']['solved']/stats['boards']['total']:4.1f}% | {100*stats['pixels']['solved']/stats['pixels']['total']:4.1f}% pixels | in {time_taken//60:.0f}:{time_taken%60:02.0f}")

    submission_df.sort_index().to_csv('submission.csv')

    exact_submission.sort_index().to_csv('submission_exact.csv')

    return submission_df

ouroborors_dataframe(test_df)
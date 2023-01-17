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









# For generating large quantities of data, we can use joblib Parallel() to take advantage of all 4 CPUs available in a Kaggle Notebook

life_step = life_step_njit

def life_steps(boards: List[np.ndarray]) -> List[np.ndarray]:

    """ Parallel version of life_step() but for an array of boards """

    return Parallel(-1)( delayed(life_step)(board) for board in boards )
# RULES: https://www.kaggle.com/c/conway-s-reverse-game-of-life/data

def generate_random_board() -> np.ndarray:

    # An initial board was chosen by filling the board with a random density between 1% full (mostly zeros) and 99% full (mostly ones).

    # DOCS: https://cmdlinetips.com/2019/02/how-to-create-random-sparse-matrix-of-specific-density/

    density = np.random.random() * 0.98 + 0.01

    board   = scipy.sparse.random(25, 25, density=density, data_rvs=np.ones).toarray()



    # The starting board's state was recorded after the 5 "warmup steps". These are the values in the start variables.

    for t in range(5):

        board = life_step(board)

        if np.count_nonzero(board) == 0: return generate_random_board()  # exclude empty boards and try again

    return board



def generate_random_boards(count) -> List[np.ndarray]:

    generated_boards = Parallel(-1)( delayed(generate_random_board)() for _ in range(count) )

    return generated_boards
# First check if we have a GPU available 

# __file__ is implictly defined when running on localhost, but needs to be manually set when running inside a Kaggle Notebook   

import torch

device   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

__file__ = './notebook.ipynb'
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

                print(f'{self.__class__.__name__}.load(): model has changed dimensions, discarding saved weights\n')

                pass



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

import torch

import torch.nn as nn

import torch.nn.functional as F





class GameOfLifeForward(GameOfLifeBase):

    """

    This implements the life_step() function as a Neural Network function

    Training/tested to 100% accuracy over 10,000 random boards

    """

    def __init__(self):

        super().__init__()

        self.dropout = nn.Dropout(p=0.0)  # disabling dropout as it causes excessive (hour+) runtimes to reach 100% accuracy 



        # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board | with dropout=0.1

        # Finished Training: GameOfLifeForward - 240995 epochs in 3569.1s

        self.conv1   = nn.Conv2d(in_channels=1,     out_channels=128, kernel_size=(3,3), padding=1, padding_mode='circular')

        self.conv2   = nn.Conv2d(in_channels=128,   out_channels=16,  kernel_size=(1,1))

        self.conv3   = nn.Conv2d(in_channels=16,    out_channels=8,   kernel_size=(1,1))

        self.conv4   = nn.Conv2d(in_channels=1+8,   out_channels=1,   kernel_size=(1,1))

        self.apply(self.weights_init)





    def forward(self, x):

        x = input = self.cast_inputs(x)



        x = self.conv1(x)

        x = F.leaky_relu(x)



        x = self.conv2(x)

        x = F.leaky_relu(x)

        x = self.dropout(x)



        x = self.conv3(x)

        x = F.leaky_relu(x)

        x = self.dropout(x)



        x = torch.cat([ x, input ], dim=1)  # remind CNN of the center cell state before making final prediction

        x = self.conv4(x)

        x = torch.sigmoid(x)



        return x
import atexit

import sys

import time



import numpy as np

import torch

import torch.optim as optim

from torch import tensor





def train(model, batch_size=25, l1=0, l2=0, timeout=0, reverse_input_output=False):

    print(f'Training: {model.__class__.__name__}')

    time_start = time.perf_counter()



    atexit.register(model.save)      # save on exit - BrokenPipeError doesn't trigger finally:

    model.load().train().unfreeze()  # enable training and dropout



    # NOTE: criterion loss function now defined via model.loss()

    optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)



    # epoch: 14481 | board_count: 362000 | loss: 0.0000385726 | accuracy = 0.9990336000 | time: 0.965ms/board

    scheduler = None



    # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board

    # Finished Training: GameOfLifeForward - 240995 epochs in 3569.1s

    scheduler_config = {

        'optimizer': optimizer,

        'max_lr':       1e-2,

        'base_lr':      1e-4,

        'step_size_up': 250,

        'mode':         'exp_range',

        'gamma':        0.8

    }

    scheduler = torch.optim.lr_scheduler.CyclicLR(**scheduler_config)



    success_count = 10_000

    

    epoch        = 0

    board_count  = 0

    last_loss    = np.inf

    loop_loss    = 0

    loop_acc     = 0

    loop_count   = 0

    epoch_losses     = [last_loss]

    epoch_accuracies = [ 0 ]

    num_params = torch.sum(torch.tensor([

        torch.prod(torch.tensor(param.shape))

        for param in model.parameters()

    ]))

    try:

        for epoch in range(1, sys.maxsize):

            if np.min(epoch_accuracies[-success_count//batch_size:]) == 1.0:   break  # multiple epochs of 100% accuracy to pass

            if timeout and timeout < time.perf_counter() - time_start:  break

            epoch_start = time.perf_counter()



            inputs_np   = [ generate_random_board() for _     in range(batch_size) ]

            expected_np = [ life_step(board)        for board in inputs_np         ]

            inputs      = model.cast_inputs(inputs_np).to(device)

            expected    = model.cast_inputs(expected_np).to(device)



            # This is for GameOfLifeReverseOneStep() function, where we are trying to learn the reverse function

            if reverse_input_output:

                inputs_np, expected_np = expected_np, inputs_np

                inputs,    expected    = expected,    inputs

                assert np.all( life_step(expected_np[0]) == inputs_np[0] )





            optimizer.zero_grad()

            outputs = model(inputs)

            loss    = model.loss(outputs, expected, inputs)

            if l1 or l2:

                l1_loss = torch.sum(tensor([ torch.sum(torch.abs(param)) for param in model.parameters() ])) / num_params

                l2_loss = torch.sum(tensor([ torch.sum(param**2)         for param in model.parameters() ])) / num_params

                loss   += ( l1_loss * l1 ) + ( l2_loss * l2 )



            loss.backward()

            optimizer.step()

            if scheduler is not None:

                # scheduler.step(loss)  # only required for

                scheduler.step()



            # noinspection PyTypeChecker

            last_accuracy = model.accuracy(outputs, expected, inputs)  # torch.sum( outputs.to(torch.bool) == expected.to(torch.bool) ).cpu().numpy() / np.prod(outputs.shape)

            last_loss     = loss.item() / batch_size



            epoch_losses.append(last_loss)

            epoch_accuracies.append( last_accuracy )



            loop_loss   += last_loss

            loop_acc    += last_accuracy

            loop_count  += 1

            board_count += batch_size

            epoch_time   = time.perf_counter() - epoch_start

            time_taken   = time.perf_counter() - time_start



            # Print statistics after each epoch

            if( (epoch <= 10)

             or (np.log10(board_count) % 1 == 0)

             or (board_count <  10_000 and board_count %  1_000 == 0)  

             or (board_count < 100_000 and board_count % 10_000 == 0) 

             or (board_count % 100_000 == 0)

            ):

                print(f'epoch: {epoch:6d} | board_count: {board_count:7d} | loss: {loop_loss/loop_count:.10f} | accuracy = {loop_acc/loop_count:.10f} | time: {1000*epoch_time/batch_size:.3f}ms/board | {time_taken//60:2.0f}m {time_taken%60:02.0f}s')

                loop_loss  = 0

                loop_acc   = 0

                loop_count = 0

        print(f'Successfully trained to 100% accuracy over the last {success_count} boards')

        print(f'epoch: {epoch:6d} | board_count: {board_count:7d} | loss: {np.mean(epoch_losses[-success_count//batch_size:]):.10f} | accuracy = {np.min(epoch_accuracies[-success_count//batch_size:]):.10f} | time: {1000*epoch_time/batch_size:.3f}ms/board')

                

    except (BrokenPipeError, KeyboardInterrupt):

        pass

    except Exception as exception:

        print(exception)

        raise exception

    finally:

        time_taken = time.perf_counter() - time_start

        print(f'Finished Training: {model.__class__.__name__} - {epoch} epochs in {time_taken:.1f}s')

        model.save()

        atexit.unregister(model.save)   # model now saved, so cancel atexit handler

        # model.eval()                  # disable dropout
# !rm ./models/GameOfLifeForward.pth

model = GameOfLifeForward()

train(model)
def test_GameOfLifeForward_single():

    """

    Test GameOfLifeForward().predict() works with single board datastructure semantics

    """

    model = GameOfLifeForward()



    input    = generate_random_board()

    expected = life_step(input)

    output   = model.predict(input)

    assert np.all( output == expected )  # assert 100% accuracy







def test_GameOfLifeForward_batch(count=1000):

    """

    Test GameOfLifeForward().predict() also works batch-mode datastructure semantics

    Batch mode is limited to only 1000 boards at a time, which prevents CUDA out of memory exceptions

    """

    model = GameOfLifeForward()



    # As well as in batch mode

    for _ in range(max(1,count//1000)):

        inputs   = generate_random_boards(1000)

        expected = life_steps(inputs)

        outputs  = model.predict(inputs)

        assert np.all( outputs == expected )  # assert 100% accuracy
# GameOfLifeForward can successfully predict a million boards in a row correctly

time_start = time.perf_counter()

test_GameOfLifeForward_single()

test_GameOfLifeForward_batch(1_000_000)

time_taken = time.perf_counter() - time_start

print(f'All tests passed! ({time_taken:.0f}s)')
# Discovery:

# neural networks in batch mode can be faster than C compiled classical code, but slower when called in a loop

# also scipy uses numba @njit under the hood

#

# number: 1 | batch_size = 1000

# life_step() - numpy         = 181.7µs

# life_step() - scipy         = 232.1µs

# life_step() - njit          = 433.5µs   # includes @njit compile times

# gameOfLifeForward() - loop  = 567.7µs

# gameOfLifeForward() - batch = 1807.5µs  # includes .pth file loadtimes

#

# number: 100 | batch_size = 1000

# gameOfLifeForward() - batch =  29.8µs  # faster than even @njit or scipy

# life_step() - scipy         =  35.6µs

# life_step() - njit          =  42.8µs

# life_step() - numpy         = 180.8µs

# gameOfLifeForward() - loop  = 618.3µs  # much slower, but still fast compared to an expensive function



def profile_GameOfLifeForward():

    import timeit

    import operator



    model  = GameOfLifeForward().load().to(device)

    for batch_size in [1, 10_000]:

        boards = generate_random_boards(batch_size)

        number = 1

        timings = {

            'gameOfLifeForward() - batch': timeit.timeit(lambda:   model(boards),                                number=number),

            'gameOfLifeForward() - loop':  timeit.timeit(lambda: [ model(board)           for board in boards ], number=number),

            'life_step() - njit':          timeit.timeit(lambda: [ life_step_njit(board)  for board in boards ], number=number),

            'life_step() - numpy':         timeit.timeit(lambda: [ life_step_numpy(board) for board in boards ], number=number),

            'life_step() - scipy':         timeit.timeit(lambda: [ life_step_scipy(board) for board in boards ], number=number),

        }

        print(f'{device} | batch_size = {len(boards)}')

        for key, value in sorted(timings.items(), key=operator.itemgetter(1)):

            print(f'- {key:27s} = {value/number/len(boards) * 1_000:7.3f}ms')

        print()
device = torch.device("cpu")

profile_GameOfLifeForward()



if torch.cuda.is_available():

    device = torch.device("cuda:0")

    profile_GameOfLifeForward()
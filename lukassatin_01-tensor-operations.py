# Experimental Kaggle TPU support for PyTorch (should work also on Google Collab)!

#!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

#!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
# Import torch and other required modules

import torch

import random

import os

import numpy as np

from tabulate import tabulate

import matplotlib.pyplot as plt
def seed_everything(seed):

    """

    Seeds basic parameters for reproductibility of results

    

    Arguments:

        seed {int} -- Number of the seed

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
seed = 2020

seed_everything(seed)
# Lazily set the current device to reuse in whole project (important!)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# If GPU, prints: 'cuda:0', where 0 is GPU id (for multi GPU system)

# If CPU, prints: 'cpu'

# If TPU, prints: 'cpu' as well

print(device)



# Additional GPU nice to have info

if torch.cuda.is_available():

    # How to get current GPU

    print(torch.cuda.get_device_name())

    print(torch.cuda.get_device_properties(torch.cuda.current_device()))

    

    # How to navigate on multi GPU system

    for i in range(0, torch.cuda.device_count()):

        print(torch.cuda.get_device_properties(i))
# Now we initialise Tensor, it is important to supply device parameter everywhere in order to make your code portable!

tensor = torch.tensor(range(15), device=device)
# Now if we try to print Tensor as a Numpy array, it will break on GPU

print("Print from Numpy directly:")

print(tensor.numpy())

print()

# The solution is to move it to CPU memory

print("A better portable solution for multi-platform development (GPU):")

print(tensor.cpu().numpy())
# Example 1 - Print 1D Tensor

tensor_1d = torch.tensor([1.1, 2., 3., 4., 5.], device=device)



torch.set_printoptions(profile='default', sci_mode=False)



print('Python default:')

print(tensor_1d)

print()

print('PyTorch Scientific Mode:')

torch.set_printoptions(profile='short', sci_mode=True)

print(tensor_1d)

torch.set_printoptions(profile='default', sci_mode=False)
# Example 2 - Convert 1D Tensor to 2D Tensor

tensor_2d = tensor_1d.view(tensor_1d.size()[0], 1)



print(tabulate(tensor_2d))
# Example 3 - breaking it

# for multi-dimensional tensors we can use charting library to plot them

plt.imshow(tensor_2d)



plt.imshow(tensor_2d.cpu()) # do not forget for cpu() method call as described in Section 1
# Example 1 - working

import numpy as np

import IPython.core.display



def _html_repr_helper(contents, index, is_horz):

    dims_left = contents.ndim - len(index)

    if dims_left == 0:

        s = contents[index]

    else:

        s = '<span class="numpy-array-comma">,</span>'.join(

            _html_repr_helper(contents, index + (i,), is_horz) for i in range(contents.shape[len(index)])

        )

        s = ('<span class="numpy-array-bracket numpy-array-bracket-open">[</span>'

            '{}'

            '<span class="numpy-array-bracket numpy-array-bracket-close">]</span>'.format(s))

        

    # apply some classes for styling

    classes = []

    classes.append('numpy-array-slice')

    classes.append('numpy-array-ndim-{}'.format(len(index)))

    classes.append('numpy-array-ndim-m{}'.format(dims_left))

    if is_horz(contents, len(index)):

        classes.append('numpy-array-horizontal')

    else:

        classes.append('numpy-array-vertical')

    

    hover_text = '[{}]'.format(','.join('{}'.format(i) for i in (index + (':',) * dims_left)))



    return "<span class='{}' title='{}'>{}</span>".format(

        ' '.join(classes), hover_text, s,

    )



basic_css = """

    .numpy-array {

        display: inline-block;

    }

    .numpy-array .numpy-array-slice {

        border: 1px solid #cfcfcf;

        border-radius: 4px;

        margin: 1px;

        padding: 1px;

        display: flex;

        flex: 1;

        text-align: right;

        position: relative;

    }

    .numpy-array .numpy-array-slice:hover {

        border: 1px solid #66BB6A;

    }

    .numpy-array .numpy-array-slice.numpy-array-vertical {

        flex-direction: column;

    }

    .numpy-array .numpy-array-slice.numpy-array-horizontal {

        flex-direction: row;

    }

    .numpy-array .numpy-array-ndim-m0 {

        padding: 0 0.5ex;

    }

    

    /* Hide the comma and square bracket characters which exist to help with copy paste */

    .numpy-array .numpy-array-bracket {

        font-size: 0;

        position: absolute;

    }

    .numpy-array span .numpy-array-comma {

        font-size: 0;

        height: 0;

    }

"""



show_brackets_css = """

    .numpy-array.show-brackets .numpy-array-slice {

        border-radius: 0;

    }

    .numpy-array.show-brackets .numpy-array-bracket {

        border: 1px solid black; 

        border-radius: 0;  /* looks better without... */

    }

    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-open {

        top: -1px;

        bottom: -1px;

        left: -1px;

        width: 10px;

        border-right: none;

        border-top-right-radius: 0;

        border-bottom-right-radius: 0;

    }

    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-close {

        top: -1px;

        bottom: -1px;

        right: -1px;

        width: 10px;

        border-left: none;

        border-top-left-radius: 0;

        border-bottom-left-radius: 0;

    }

    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-open {

        top: -1px;

        right: -1px;

        left: -1px;

        height: 10px;

        border-bottom: none;

        border-bottom-right-radius: 0;

        border-bottom-left-radius: 0;

    }

    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-close {

        left: -1px;

        bottom: -1px;

        right: -1px;

        height: 10px;

        border-top: none;

        border-top-right-radius: 0;

        border-top-left-radius: 0;

    }

"""



def make_pretty(self, show_brackets=False, is_horz=lambda arr, ax: ax == arr.ndim - 1):



    classes = ['numpy-array']

    css = basic_css

    if show_brackets:

        classes += ['show-brackets']

        css += show_brackets_css

    return IPython.core.display.HTML(

        """<style>{}</style><div class='{}'>{}</div>""".format(

            css,

            ' '.join(classes),

            _html_repr_helper(self, (), is_horz))

    )
tensor_3d = torch.rand(10, 1, 3)



print(tabulate(tensor_3d, showindex='always', tablefmt='pretty'))
# Example 2 - working

make_pretty(tensor_3d) # make sure this call goes after print()
# Example 1 - working

import plotly.offline as py

import plotly.graph_objs as go



py.init_notebook_mode(connected=True)



# Temperature: -40 to +85°C

# Humidity: 0-100%

# Pressure: 300-1100 hPa

    

temperature = [-10, 0, 5, 15, 30]

humidity = [20, 40, 40, 60, 70]

pressure = [400, 500, 600, 700, 800]



weather_tensor = torch.tensor([temperature, pressure, humidity], device=device)



# temperature on x axis to represent height in graph

z, y, x = weather_tensor



fig = go.Figure(data=[

    go.Mesh3d(

        x=x,

        y=y,

        z=z,

        colorbar_title='Temperature',

        colorscale=[[0, 'gold'],

                    [0.5, 'mediumturquoise'],

                    [1, 'magenta']],

        # Intensity of each vertex, which will be interpolated and color-coded

        intensity=z,

        showscale=True

    )

])



fig.update_layout(

    title="Weather Mesh Example",

    font=dict(

        family="Courier New, monospace",

        size=16,

        color="#7f7f7f"

    )

)



camera = dict(

    eye=dict(x=-2, y=-2, z=0.1)

)



fig.update_layout(scene_camera=camera)



fig.show()

# Example 1 - normalizing data

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()



# first reshape (-1, 1) to change list dimension from 1D to 2D

temperature_normalized = min_max_scaler.fit_transform(weather_tensor[0].numpy().reshape(-1, 1))



# now reshape the list again to single values

temperature_normalized = temperature_normalized.reshape(-1)



print(tabulate([weather_tensor[0].numpy(), temperature_normalized], showindex='always', tablefmt='pretty'))
# Example 2 - filter or mask tensor data

tensor_x = torch.tensor([0.1, 0.5, -1.0, 0, 1.2, 0])



print(tensor_x)



mask = tensor_x >= 0 # This is the important step, where you define filtering condition



print(mask)



indices = torch.nonzero(mask)



print(tensor_x[indices]) # mapping index example



print(indices) # index from original tensor
# Example 3 - basic gather example with matrix

t = torch.tensor([[1,2],[3,4]])

torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
def get_splits(x):

    bins = np.arange(0, np.ceil(x[:,1].max())+1)

    d = torch.from_numpy(np.digitize(x.numpy()[:, 1], bins))

    _, counts = torch.unique(d, return_counts=True)

    return torch.split(x, counts.tolist())



# create tensor

N = 50

c0 = torch.randn(N)

c1 = torch.arange(N) + 0.1 * torch.randn(N)

x = torch.stack((c0, c1), dim=1)



print(*get_splits(x), sep='\n')



print(tabulate(get_splits(x), showindex='always', tablefmt='pretty'))
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project="01-tensor-operations", environment=None)
# from ca_utils import display_vid, plot_task, load_data
import numpy as np

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw


import os

try:
    import moviepy
except:
    print('installing moviepy')
    os.system('pip install moviepy')

from moviepy.editor import ImageSequenceClip


from pathlib import Path
    
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')


cmap_lookup = [
        '#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap_lookup = [ np.array( [int(x[1:3],16), int(x[3:5],16), int(x[5:],16)])  for x in cmap_lookup]


def cmap(x):
    """ 
        Translate a task matrix to a color coded version
        
        arguments
            x : a h x w task matrix
        returns 
            a h x w x 3 matrix with colors instead of numbers
    """
    y = np.zeros((*x.shape, 3))
    y[x<0, :] = np.array([112,128,144])
    y[x>9,:] = np.array([255,248,220])
    for i, c in enumerate(cmap_lookup):        
        y[x==i,:] = c
    return y
    
def draw_one(x, k=20):
    """
        Create a PIL image from a task matrix, the task will be 
        drawn using the default color coding with grid lines
        
        arguments
            x : a task matrix
            k = 20 : an up scaling factor
        returns
            a PIL image 
            
    """
    img = Image.fromarray(cmap(x).astype(np.uint8)).resize((x.shape[1]*k, x.shape[0]*k), Image.NEAREST )
    
    draw = ImageDraw.Draw(img)
    for i in range(x.shape[0]):
        draw.line((0, i*k, img.width, i*k), fill=(80, 80, 80), width=1)   
    for j in range(x.shape[1]):    
        draw.line((j*k, 0, j*k, img.height), fill=(80, 80, 80), width=1)
    return img


def vcat_imgs(imgs, border=10):
    """
        Concatenate images vertically
        
        arguments:
            imgs : an array of PIL images
            border = 10 : the size of space between images
        returns:
            a PIL image
    """
    
    h = max(img.height for img in imgs)
    w = sum(img.width for img in imgs)
    res_img = Image.new('RGB', (w + border*(len(imgs)-1), h), color=(255, 255, 255))

    offset = 0
    for img in imgs:
        res_img.paste(img, (offset,0))
        offset += img.width + border
        
    return res_img




def plot_task(task):
    """
        Plot a task
        
        arguments:
            task : either a task read with `load_data` or a task name
    """
    
    if isinstance(task, str):
        task_path = next( data_path / p /task for p in ('training', 'evaluation','test') if (data_path / p / task).exists() )
        task = load_data(task_path)
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(n*4, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    def go(ax, title, x):
        ax.imshow(draw_one(x), interpolation='nearest')
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        
    for i, t in enumerate(task["train"]):
        go(axs[0][fig_num], f'Train-{i} in', t["input"])
        go(axs[1][fig_num], f'Train-{i} out', t["output"])
        fig_num += 1
    for i, t in enumerate(task["test"]):
        go(axs[0][fig_num], f'Test-{i} in', t["input"])
        go(axs[1][fig_num], f'Test-{i} out', t["output"])
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    
    
    
def trace_automata(step_fn, input, n_iter, n_hidden, loadbar=True):
    """
        Execute an automata and return all the intermediate states
        
        arguments:
            step_fn : transition rule function, should take two arguments `input` and `hidden_i`, 
                should return an output grid an a new hidden hidden grid
            n_iter : num of iteration to perform
            n_hidden: number of hidden grids, if set to 0 `hidden_i` will be set to None
            laodbar = True: weather display loadbars
        returns:
            an array of tuples if output and hidden grids
    """
    
    hidden = np.zeros((n_hidden, *input.shape)) if n_hidden > 0 else None
    
    trace = [(input, hidden)]
    
    its = range(n_iter)
    if loadbar:
        its = tqdm(its, desc='Step')
    for _ in its:
        output, hidden = step_fn(input, hidden)
        trace.append((output, hidden))        
        input = output
    return trace


def vis_automata_trace(states, loadbar=True, prefix_image=None): 
    """
        Create a video from an array of automata states
        
        arguments:
            states : array of automata steps, returned by `trace_automata()`
            loadbar = True: weather display loadbars
            prefix_image = None: image to add to the beginning of each frame 
        returns 
            a moviepy ImageSequenceClip
    """
    frames = []
    if loadbar:
        states = tqdm(states, desc='Frame')
    for i, (canvas, hidden) in enumerate(states):
        
        frame = []
        if prefix_image is not None:
            frame.append(prefix_image)
        frame.append(draw_one(canvas))
        if hidden is not None:
            frame.extend(draw_one(h) for h in hidden)
        frames.append(vcat_imgs(frame))            
        
    return ImageSequenceClip(list(map(np.array, frames)), fps=10)


from moviepy.editor import clips_array, CompositeVideoClip


from moviepy.video.io.html_tools import html_embed, HTML2

def display_vid(vid, verbose = False, **html_kw):
    """
        Display a moviepy video clip, useful for removing loadbars 
    """
    
    rd_kwargs = { 
        'fps' : 10, 'verbose' : verbose 
    }
    
    if not verbose:
         rd_kwargs['logger'] = None
    
    return HTML2(html_embed(vid, filetype=None, maxduration=60,
                center=True, rd_kwargs=rd_kwargs, **html_kw))


def vis_automata_task(tasks, step_fn, n_iter, n_hidden, vis_only_ix=None):
    """
        Visualize the automata steps during the task solution
        arguments:
            tasks : the task to be solved by the automata
            step_fn : automata transition function as passed to `trace_automata()`
            n_iter : number of iterations to perform
            n_hidden : number of hidden girds
    """
    
    n_vis = 0        
    
    def go(task, n_vis):
        
        if vis_only_ix is not None and vis_only_ix != n_vis:
            return 
        
        trace = trace_automata(step_fn, task['input'], n_iter, n_hidden, loadbar=False)
        vid = vis_automata_trace(trace, loadbar=False, prefix_image=draw_one(task['output']))
        display(display_vid(vid))
        
    
        
    for task in (tasks['train']):
        n_vis += 1
        go(task, n_vis)
        
    for task in (tasks['test']):
        n_vis += 1
        go(task, n_vis)
    
    
#
# Data IO
#

import os
import json

training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

def load_data(p, phase=None):
    """
        Load task data
        
    """
    if phase in {'training', 'test', 'evaluation'}:
        p = data_path / phase / p
    
    task = json.loads(Path(p).read_text())
    dict_vals_to_np = lambda x: { k : np.array(v) for k, v in x.items() }
    assert set(task) == {'test', 'train'}
    res = dict(test=[], train=[])
    for t in task['train']:
        assert set(t) == {'input', 'output'}
        res['train'].append(dict_vals_to_np(t))
    for t in task['test']:
        assert set(t) == {'input', 'output'}
        res['test'].append(dict_vals_to_np(t))
        
    return res
task = load_data('08ed6ac7.json', phase='training')
plot_task(task)
from moviepy.editor import VideoFileClip

display_vid(VideoFileClip('../input/ca-videos/IaANIHh6.mp4'), loop=True, autoplay=True)
from ca_utils import vis_automata_task
import numpy as np
from itertools import product

nbh = lambda x, i, j: { 
    (ip, jp) : x[i+ip, j+jp] 
        for ip, jp in product([1, -1, 0], repeat=2) 
            if 0 <= i+ip < x.shape[0] and 0 <= j+jp < x.shape[1]
}
import matplotlib.colors as colors
import matplotlib.pyplot as plt

_cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=_cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()
task = load_data('db3e9e38.json', phase='training')

plot_task(task)

def compute_db3e9e38_part_automata_step(input, hidden_i):
    #
    # this function takes as input the input grid and outputs the output grid

    # input, is the input grid
    # ignore hidden_i for now
    
    # for convenience let's name the colors
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    # let's create the output grid
    output = np.zeros_like(input, dtype=int)
    
    # here we iterate over all the coordinates of cells in the input grid
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        # current cell and current neighborhood 
        i_c = input[i, j]                
        i_nbh = nbh(input, i, j)
        
        # Here our transition rule starts 
        # R1: if the current call has color
        if i_c != blk:
            output[i, j] = i_c # leave it there 
            
        # R2: if the current cell is black and as it's left neighbor and lower left neighbors are orange cells 
        elif i_c == blk and i_nbh.get((0, -1)) == orn and i_nbh.get((1, -1)) == orn:
            output[i, j] = azu # paint it in light blue (azure)
        
        # R3: if the current cell is black and as it's left neighbor and lower left neighbors are light blue cells  
        elif i_c == blk and i_nbh.get((0, -1)) == azu and i_nbh.get((1, -1)) == azu:
            output[i, j] = orn # paint it in orange
        
    return output, hidden_i
            
vis_automata_task(task, compute_db3e9e38_part_automata_step, 16, 0)
def compute_db3e9e38_automata_step(input, hidden_i):
    # ignore hidden_i for now
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    output = np.zeros_like(input, dtype=int)
    
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]                
        i_nbh = nbh(input, i, j)
        
        # R1
        if i_c != blk:
            output[i, j] = i_c
            
        # R2 and it's symmetrical variant
        if i_c == blk and i_nbh.get((0, 1)) == orn and i_nbh.get((1, 1)) == orn:
            output[i, j] = azu
        elif i_c == blk and i_nbh.get((0, -1)) == orn and i_nbh.get((1, -1)) == orn:
            output[i, j] = azu
        
        # R3 and it's symmetrical variant
        elif i_nbh.get((0, 1)) == azu and i_nbh.get((1, 1)) == azu:
            output[i, j] = orn
        elif i_nbh.get((0, -1)) == azu and i_nbh.get((1, -1)) == azu:
            output[i, j] = orn
        
    return output, hidden_i
            
        

vis_automata_task(task, compute_db3e9e38_automata_step, 16, 0)
task = load_data('b27ca6d3.json', phase='training')

plot_task(task)
def compute_b27ca6d3_part3_automata_step(input, hidden_i):
    # ignore hidden_i for now
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    output = np.zeros_like(input, dtype=int)
    
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]       
        i_nbh = nbh(input, i, j)
        
        is_top_b, is_bottom_b = i == 0, i == input.shape[0]-1
        is_left_b, is_right_b = j == 0, j == input.shape[1]-1
        is_b = is_top_b or is_bottom_b or is_left_b or is_right_b
        
        # clock wise orderd neighboring elements
        cw_nbh_ixs = [ (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        # consecutive cell pairs
        pair_ixs = list(zip(cw_nbh_ixs[:-1], cw_nbh_ixs[1:])) + [(cw_nbh_ixs[-1],cw_nbh_ixs[0])]
                    
        # sides of the rectangle formed by the neighboring cells
        side_ixs = [ 
            cw_nbh_ixs[:3], # top
            cw_nbh_ixs[2:5], # left
            cw_nbh_ixs[4:7], # bottom 
            cw_nbh_ixs[6:] + cw_nbh_ixs[:1] # right
        ]        
        
        # tests if all the cells are non border ones
        all_present = lambda s1, c, s2: all(x in i_nbh for x in [s1, c, s2])
        # tests if the three cells are colored green, red, green
        test_side = lambda s1, c, s2: (i_nbh.get(s1, grn),i_nbh.get(c, red),i_nbh.get(s2, grn)) == (grn, red, grn)
        
        # corners of the square formed by the neighboring pixels
        corner_ixs = [ 
            cw_nbh_ixs[1:4], # top right
            cw_nbh_ixs[3:6], # bottom right
            cw_nbh_ixs[5:8], # bottom left
            cw_nbh_ixs[7:] + cw_nbh_ixs[:2] # top left
        ]
        
        # R0 if cell has color 
        if i_c != blk:
            output[i, j] = i_c # do nothing 
               
        # R1: if the neighborhood contains two consecutive red cells 
        elif any(i_nbh.get(ix1) == red and i_nbh.get(ix2) == red for ix1, ix2 in pair_ixs):
            output[i, j] = grn   # color in green
            
        # R2: if the neighborhood contains three consecutive cells colored with green red green 
        elif any( test_side(s1, c, s2) for s1, c, s2 in side_ixs if all_present( s1, c, s2)):
            output[i, j] = grn # color in green 
        
        # R3: if the neighborhood contains three consecutive cells colored with green red green arranged in a corner
        elif  any( test_side(s1, c, s2) for s1, c, s2 in corner_ixs if all_present(s1, c, s2)):
            output[i, j] = grn 
        
        
    return output, hidden_i

vis_automata_task(task, compute_b27ca6d3_part3_automata_step, 5, 0)
def compute_b27ca6d3_automata_step(input, hidden_i):
    
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    output = np.zeros_like(input, dtype=int)
    
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]       
        i_nbh = nbh(input, i, j)
        
        is_top_b, is_bottom_b = i == 0, i == input.shape[0]-1
        is_left_b, is_right_b = j == 0, j == input.shape[1]-1
        is_b = is_top_b or is_bottom_b or is_left_b or is_right_b
        
        # clock wise orderd neiborhood elements
        cw_nbh_ixs = [ (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        # consecutive cell pairs
        pair_ixs = list(zip(cw_nbh_ixs[:-1], cw_nbh_ixs[1:])) + [(cw_nbh_ixs[-1],cw_nbh_ixs[0])]
                    
        # sides of the rectangle formed by the neighboring cells
        side_ixs = [ 
            cw_nbh_ixs[:3], # top
            cw_nbh_ixs[2:5], # left
            cw_nbh_ixs[4:7], # bottom 
            cw_nbh_ixs[6:] + cw_nbh_ixs[:1] # right
        ]        
        
        # tests if all the cells are non border ones
        all_present = lambda s1, c, s2: all(x in i_nbh for x in [s1, c, s2])
        # tests if the three cells are colored green, red, green
        test_side = lambda s1, c, s2: (i_nbh.get(s1, grn),i_nbh.get(c, red),i_nbh.get(s2, grn)) == (grn, red, grn)
        # tests if the center cell is present and at least one on the side
        some_present = lambda s1, c, s2: c in i_nbh and (s1 in i_nbh or s2 in i_nbh)
        
        # corners of the square formed by the neighboring pixels
        corner_ixs = [ 
            cw_nbh_ixs[1:4], # top right
            cw_nbh_ixs[3:6], # bottom right
            cw_nbh_ixs[5:8], # bottom left
            cw_nbh_ixs[7:] + cw_nbh_ixs[:2] # top left
        ]
        
        
        # R0 if cell has color 
        if i_c != blk:
            output[i, j] = i_c # do nothing 
        # R1: if the neighborhood contains two consecutive red cells 
        elif any(i_nbh.get(ix1) == red and i_nbh.get(ix2) == red for ix1, ix2 in pair_ixs):
            output[i, j] = grn   # color in green

        # R2: if the neighborhood contains three consecutive cells colored with green red green 
        elif i_c == blk and any( test_side(s1, c, s2) for s1, c, s2 in side_ixs if all_present( s1, c, s2)):
            output[i, j] = grn # color in green 
            
        # R3: if the neighborhood contains three consecutive cells colored with green red green arranged in a corner
        elif i_c == blk and any( test_side(s1, c, s2) for s1, c, s2 in corner_ixs if all_present(s1, c, s2)):
            output[i, j] = grn 
            
        # R4+: if we are near a border and one green and one red consecutive cells are present
        elif i_c == blk and is_b and any( test_side(s1, c, s2) for s1, c, s2 in side_ixs if some_present( s1, c, s2) ):
            output[i, j] = grn 
        
        
    return output, hidden_i

vis_automata_task(task, compute_b27ca6d3_automata_step, 5, 0)
task = load_data('00d62c1b.json', phase='training')

plot_task(task)

def compute_00d62c1b_part_automata_step(input, hidden_i):
    
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    output = np.zeros_like(input, dtype=int)
    
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]
        i_nbh = nbh(input, i, j)        
        # direct neighbors to the current cell
        i_direct_nbh = { k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)} }
        
        is_top_b, is_bottom_b = i == 0, i == input.shape[0]-1
        is_left_b, is_right_b = j == 0, j == input.shape[1]-1
        is_b = is_top_b or is_bottom_b or is_left_b or is_right_b
        
        
        if i_c == grn:
            output[i, j] = grn        
        # R1: create yellow cells where a lot of green cells are
        elif sum(1 for v in i_nbh.values() if v == grn) >= 4 and red not in i_direct_nbh.values():
            output[i, j] = ylw
            
        # R3: set fire to cells near the border
        elif i_c == blk and is_b and ylw in i_direct_nbh.values():
            output[i, j] = red
        # R4: make the fire spread - color in red all yellow cells touching red ones
        elif i_c == ylw and red in i_nbh.values():
            output[i, j] = red
        
        # R2: propagate yellow cells in the empty places
        elif i_c == blk and ylw in i_direct_nbh.values():
            output[i, j] = ylw
        # R5: make the 'fire burn'
        elif i_c == red and red in i_nbh.values() or ylw not in i_direct_nbh.values():
            output[i, j] = blk
        else:
            #  R0
            output[i, j] = i_c
        
    return output, hidden_i

task = load_data('00d62c1b.json', phase='training')

vis_automata_task(task, compute_00d62c1b_part_automata_step, 128, 0)

def compute_00d62c1b_automata_step(input, hidden_i):
    
    blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)
    
    output = np.zeros_like(input, dtype=int)
    hidden_o = np.zeros_like(hidden_i, dtype=int)
    
    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]
        i_nbh = nbh(input, i, j)        
        # cells adagent to the current one 
        i_direct_nbh = { k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)} }
        
        i_h0 = hidden_i[0, i, j]
        
        is_top_b, is_bottom_b = i == 0, i == input.shape[0]-1
        is_left_b, is_right_b = j == 0, j == input.shape[1]-1
        is_b = is_top_b or is_bottom_b or is_left_b or is_right_b
        
        if i_h0 != blk:
            hidden_o[0, i, j] = i_h0
        
        if i_c == grn:
            output[i, j] = grn        
        
        # R1*: create yellow cells where a lot of green cells are
        elif sum(1 for v in i_nbh.values() if v == grn) >= 4 and red not in i_direct_nbh.values() and hidden_i[0, i, j] == 0:
            output[i, j] = ylw
        
            
        # R3*: set fite to cells near the border
        elif i_c == blk and is_b and ylw in i_direct_nbh.values():
            output[i, j] = red
            hidden_o[0, i, j] = 1
        # R4*: make the fire spread - color in red all yellow cells touching red ones
        elif i_c == ylw and red in i_nbh.values():
            output[i, j] = red
            hidden_o[0, i, j] = 1
        
        # R2: propagate yellow cells in the empty places
        elif i_c == blk and ylw in i_direct_nbh.values():
            output[i, j] = ylw
        
        elif i_c == red and red in i_nbh.values() or ylw not in i_direct_nbh.values():
            output[i, j] = blk
        else:
            output[i, j] = i_c
        
    return output, hidden_o

task = load_data('00d62c1b.json', phase='training')

vis_automata_task(task, compute_00d62c1b_automata_step, 50, 1)
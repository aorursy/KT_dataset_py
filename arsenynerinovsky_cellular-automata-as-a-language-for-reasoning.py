from ca_utils import display_vid, plot_task, load_data
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
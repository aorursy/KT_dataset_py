# preliminary imports

import numpy as np # linear algebra

from copy import copy

import math

import matplotlib.pyplot as plt
# loop limits,indexes names

lim_name = {

    'ltof': 'nb_K_out_tiles',

    'ltif': 'nb_K_in_tiles',

    'ltsy': 'nb_H_out_tiles',

    'ltsx': 'nb_W_out_tiles',

    'lof': 'K_out',

    'lif': 'K_in',

    'lsy': 'H_out',

    'lsx': 'W_out',

    'lfy': 'F',

    'lfx': 'F'

}

idx_name = {

    'ltof': 'mm',

    'ltif': 'nn',

    'ltsy': 'ii',

    'ltsx': 'jj',

    'lof': 'm',

    'lif': 'n',

    'lsy': 'i',

    'lsx': 'j',

    'lfy': 'ui',

    'lfx': 'uj'

}
# See A. Stoutchinin, F. Conti, L. Benini [arXiv:1902.01492] for the model



import numpy as np



def carry(array, loop, order=None):

    if array == 'y':

        if loop=='lfx':

            return True

        elif loop=='lfy':

            return True

        elif loop=='lsx':

            return False

        elif loop=='lsy':

            return False

        elif loop=='lif':

            return True

        elif loop=='lof':

            return False

    elif array == 'x':

        if loop=='lfx':

            if order.index('lfx') < order.index('lfy'):

                return True

            else:

                return False

        elif loop=='lfy':

            if order.index('lfx') < order.index('lfy'):

                return False

            else:

                return True

        elif loop=='lsx':

            return True

        elif loop=='lsy':

            return True

        elif loop=='lif':

            return False

        elif loop=='lof':

            return False

    elif array == 'w':

        if loop=='lfx':

            return False

        elif loop=='lfy':

            return False

        elif loop=='lsx':

            return True

        elif loop=='lsy':

            return True

        elif loop=='lif':

            return False

        elif loop=='lof':

            return False

    return False

    

def reuse_distance(array, loop, lim, order=('lof','lif','lsy','lsx','lfy','lfx')):

    if array == 'y':

        F = lim['lfx']

        K_in = lim['lif']

        if loop=='lfx':

            return F

        elif loop=='lfy':

            return F

        elif loop=='lsx':

            return 1

        elif loop=='lsy':

            return 1

        elif loop=='lif':

            return K_in

        elif loop=='lof':

            return 1

    elif array == 'x':

        F = lim['lfx']

        K_out = lim['lof']

        if loop=='lfx':

            if order.index('lfx') < order.index('lfy'):

                return 1

            else:

                return F

        elif loop=='lfy':

            if order.index('lfx') < order.index('lfy'):

                return F

            else:

                return 1

        elif loop=='lsx':

            if order.index('lsx') < order.index('lsy'):

                return 1

            else:

                return F

        elif loop=='lsy':

            if order.index('lsx') < order.index('lsy'):

                return F

            else:

                return 1

        elif loop=='lif':

            return 1

        elif loop=='lof':

            return K_out

    elif array == 'w':

        H_out = lim['lsy']

        W_out = lim['lsx']

        if loop=='lfx':

            return 1

        elif loop=='lfy':

            return 1

        elif loop=='lsx':

            return W_out

        elif loop=='lsy':

            return H_out

        elif loop=='lif':

            return 1

        elif loop=='lof':

            return 1

    return 1



def footprint(array, loop, lim, order=('lof','lif','lsy','lsx','lfy','lfx')):

    loop_idx = order.index(loop)

    if loop_idx == len(order)-1:

        return lim[order[loop_idx]] / reuse_distance(array, order[loop_idx], lim, order)

    else:

        return footprint(array, order[loop_idx+1], lim, order) * lim[order[loop_idx]] / reuse_distance(array, order[loop_idx], lim, order)

    

def reuse(array, loop, lim, order=('lof','lif','lsy','lsx','lfy','lfx')):

    s = 1

    try:

        loop_start_idx = order.index(loop)+1

    except ValueError:

        loop_start_idx = 0

    for i in range(loop_start_idx, len(order)):

        s *= reuse_distance(array, order[i], lim, order)

    return s

    

def traffic(array, loop, lim, order=('lof','lif','lsy','lsx','lfy','lfx')):

    lims = [ 1., ]

    try:

        loop_idx = order.index(loop) + 1

    except ValueError:

        loop_idx = 0

    for o in order:

        lims.append(lim[o])

    if loop_idx > len(order)-1:

        foot = footprint(array, order[-1], lim, order)

    else:

        foot = footprint(array, order[loop_idx], lim, order)

    return foot * np.asarray(lims)[:loop_idx+1].prod()

    

def buffer(array, loop, lim, order=('lof','lif','lsy','lsx','lfy','lfx')):

    try:

        loop_idx = order.index(loop)

        carries = carry(array, loop, order)

    except ValueError:

        loop_idx = -1

        carries = False

    if carries:

        if loop_idx == len(order)-1:

            return 1

        else:

            return footprint(array, order[loop_idx], lim, order)

    else:

        if loop_idx == len(order)-1:

            return 1

        else:

            return buffer(array, order[loop_idx+1], lim, order)

# loop-nest

def loop_nest(order=('lof','lif','lsy','lsx','lfy','lfx'), buffer_loop_w='out', buffer_loop_x='out', buffer_loop_y='out', plot=False):

    # print loop

    print("loop-nest:")

    try:

        s = ""

        for i,o in enumerate(order):

            s += "    %sfor %s in range(0, %s): # %s\n" % ("    " * i, idx_name[o], lim_name[o], o)

        print(s)

    except KeyError:

        pass

    print("reuse:")

    print("    x      = %d times" % reuse('x', buffer_loop_x, lim, order=order)) # not counting boundary in reuse calculation

    print("    w      = %d times" % reuse('w', buffer_loop_w, lim, order=order))

    print("    y/psum = %d times" % reuse('y', buffer_loop_y, lim, order=order))

    print("\nbuffer size:")

    print("    x      = %d elements" % buffer('x', buffer_loop_x, lim, order=order)) # not counting boundary in reuse calculation

    print("    w      = %d elements" % buffer('w', buffer_loop_w, lim, order=order))

    print("    y/psum = %d elements" % buffer('y', buffer_loop_y, lim, order=order))

    print("\ntraffic:")

    print("    x      = %d transfers" % traffic('x', buffer_loop_x, lim, order=order)) # not counting boundary in reuse calculation

    print("    w      = %d transfers" % traffic('w', buffer_loop_w, lim, order=order))

    print("    y/psum = %d transfers" % traffic('y', buffer_loop_y, lim, order=order))

    if plot:

        fig, ax = plt.subplots(1,3, figsize=(20,3))

        ax[0].bar(0, reuse('x', buffer_loop_x, lim, order=order), color='red')

        ax[0].bar(1, reuse('w', buffer_loop_w, lim, order=order), color='green')

        ax[0].bar(2, reuse('y', buffer_loop_y, lim, order=order), color='blue')

        ax[0].set_yscale('log')

        ax[0].set_title('reuse')

        ax[0].set_xticks((0,1,2))

        ax[0].set_xticklabels(('x','w','y'))

        ax[1].bar(0, buffer('x', buffer_loop_x, lim, order=order), color='red')

        ax[1].bar(1, buffer('w', buffer_loop_w, lim, order=order), color='green')

        ax[1].bar(2, buffer('y', buffer_loop_y, lim, order=order), color='blue')

        ax[1].set_yscale('log')

        ax[1].set_title('buffer')

        ax[1].set_xticks((0,1,2))

        ax[1].set_xticklabels(('x','w','y'))

        ax[2].bar(0, traffic('x', buffer_loop_x, lim, order=order), color='red')

        ax[2].bar(1, traffic('w', buffer_loop_w, lim, order=order), color='green')

        ax[2].bar(2, traffic('y', buffer_loop_y, lim, order=order), color='blue')

        ax[2].set_yscale('log')

        ax[2].set_title('traffic')

        ax[2].set_xticks((0,1,2))

        ax[2].set_xticklabels(('x','w','y'))

        plt.show()

lim = {

    'lof': 64, # K_out

    'lif': 64, # K_in

    'lsy': 224, # H_out

    'lsx': 224, # W_out

    'lfy': 3, # F

    'lfx': 3  # F

}

loop_nest(order=('lof','lif','lsy','lsx','lfy','lfx'), buffer_loop_w='out', buffer_loop_x='out', buffer_loop_y='out', plot=True)
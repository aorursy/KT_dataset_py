import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import cm

plt.style.use("seaborn")



!conda install -y -c bioconda forgi

!conda install -y -c bioconda viennarna



import forgi.graph.bulge_graph as fgb

import forgi.threedee.utilities.vector as ftuv

import RNA



import warnings

warnings.filterwarnings('ignore')
base = "../input/stanford-covid-vaccine/"

train = pd.read_json(base + 'train.json', lines=True)

# test = pd.read_json(base + 'test.json', lines=True)

# submission = pd.read_csv(base + 'sample_submission.csv')



train.head()
def _clashfree_annot_pos(pos, coords):

    for c in coords:

        dist = ftuv.vec_distance(c, pos)

        if dist<14:

            return False

    return True



def _find_annot_pos_on_circle(nt, coords, cg):

    for i in range(5):

        for sign in [-1,1]:

            a = np.pi/4*i*sign

            if cg.get_elem(nt)[0]=="s":

                bp = cg.pairing_partner(nt)

                anchor = coords[bp-1]

            else:

                anchor =np.mean([ coords[nt-2], coords[nt]], axis=0)

            vec = coords[nt-1]-anchor

            vec=vec/ftuv.magnitude(vec)

            rotated_vec =  np.array([vec[0]*math.cos(a)-vec[1]*math.sin(a),

                                     vec[0]*math.sin(a)+vec[1]*math.cos(a)])

            annot_pos = coords[nt-1]+rotated_vec*50

            if _clashfree_annot_pos(annot_pos, coords):

                return annot_pos

    return None

def my_plot_rna(cg, highlight_idx, df, max_val, min_val, target="reactivity", ax=None, offset=(0, 0), text_kwargs={}, backbone_kwargs={},

                nt=True):

    RNA.cvar.rna_plot_type = 1



    values = np.array(df[target].values[0])

    coords = []



    bp_string = cg.to_dotbracket_string()

    # get the type of element of each nucleotide

    el_string = cg.to_element_string()

    # i.e. eeesssshhhhsssseeee

    hl_structure_string = list(df['structure'])[0][highlight_idx]

    hl_plt_string = list(df['predicted_loop_type'])[0][highlight_idx]

    paired_string = "Paired"

    if hl_structure_string == ".":

        paired_string = "Unpaired"

        

    if ax is None:

        ax = plt.gca()



    if offset is None:

        offset = (0, 0)

    elif offset is True:

        offset = (ax.get_xlim()[1], ax.get_ylim()[1])

    else:

        pass



    vrna_coords = RNA.get_xy_coordinates(bp_string)

    # TODO Add option to rotate the plot

    for i, _ in enumerate(bp_string):

        coord = (offset[0] + vrna_coords.get(i).X,

                 offset[1] + vrna_coords.get(i).Y)

        coords.append(coord)

    coords = np.array(coords)

    # First plot backbone

    bkwargs = {"color":"gray", "zorder":0, "alpha":0.2}

    bkwargs.update(backbone_kwargs)

    ax.plot(coords[:,0], coords[:,1], **bkwargs)

    

    normalized_values = (values - min_val) / (max_val - min_val)

    # Now plot circles

    for i, coord in enumerate(coords):

        a = 1.0

        if i < 68:

            c = cm.Blues(normalized_values[i])

        else:

            c = 'black'

            a = 0.8

        circle = plt.Circle((coord[0], coord[1]), radius=8, color=c, alpha=a)



        ax.add_artist(circle)

        if cg.seq:

            text_kwargs["color"]="red"

            if nt:

                ax.annotate(cg.seq[i+1], xy=coord, ha="center", va="center", **text_kwargs )



    all_coords = list(coords)

    ntnum_kwargs = {"color":"gray"}

    ntnum_kwargs.update(text_kwargs)

    highlight_kwargs = {"color":"black", "fontsize":15}

    highlight_kwargs.update(text_kwargs)

    

    for nt in range(10, cg.seq_length, 10):

        # We try different angles

        annot_pos = _find_annot_pos_on_circle(nt, all_coords, cg)

        if annot_pos is not None:

            ax.annotate(str(nt), xy=coords[nt-1], xytext=annot_pos,

                        arrowprops={"width":1, "headwidth":1, "color":"gray"},

                        ha="center", va="center", zorder=0, **ntnum_kwargs)

            all_coords.append(annot_pos)



    annot_pos_highlight = _find_annot_pos_on_circle(highlight_idx+1, all_coords, cg)

    all_coords.append(annot_pos_highlight)

    ax.annotate("HERE!", xy=coords[highlight_idx], xytext=annot_pos_highlight,

                arrowprops={"width":1, "headwidth":1, "color":"red"},

                bbox=dict(boxstyle="round", alpha=0.1),

                ha="center", va="center", zorder=0, **highlight_kwargs)

    

    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),

                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),

               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),

                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))



    ax.set_aspect('equal', 'datalim')

    ax.update_datalim(datalim)

    ax.autoscale_view()

    ax.set_axis_off()

    

    val = str(values[highlight_idx])

    info = df['id'].values[0] + "\n" + target + ": " + val + "\nNT: " + str(cg.seq[highlight_idx+1]) \

    + "\nloop type (bpRNA): " + el_string[highlight_idx] + "(" + hl_plt_string + "), " \

    + paired_string

    ax.text(ax.get_xlim()[0], ax.get_ylim()[-1], info, fontsize=18)



    return (ax, coords)

train = train[train['SN_filter'] == 1]
def make_target(target_str):    

    target = train[['id', target_str]]

    target = target.explode(column=target_str)

    i = 0

    for _, row in target.iterrows():

        row['id'] += "_" + str(i % 68)

        i += 1

    target = target.sort_values(by=[target_str], ascending=False)

    target = target.reset_index(drop=True)

    return target
def plot_distribution(target, target_str):

    target_list = list(target[target_str])

    plt.figure(figsize=(12, 8))

    plt.suptitle(target_str + " distribution", fontsize=18)

    plt.subplot(2, 1, 1)

    plt.hist(target_list, density=True, bins="auto")

    plt.subplot(2, 1, 2)

    sns.boxplot(target_list)

    plt.xlabel(target_str)

    print(target[target_str].astype(float).describe())

    plt.show()
def plot_high_target(n, target, target_str, nt=False, fig_size=None):

    r = (n // 5) if (n // 5 > 0) else 1

    c = 5



    if fig_size is None:

        fig_size = (40, 10*r)



    fig, ax = plt.subplots(r, c, figsize=fig_size)

    plt.suptitle(f'{r*c} largest {target_str} values', fontsize=25)

    

    max_value = np.max(target[target_str])

    min_value = np.min(target[target_str])

    

    for i in range(r * c):

        nt_id = target.loc[i].id

        nt_id_num = int(nt_id.split("_")[2])

        nt_id = nt_id[:12]

        df = train[train['id'] == nt_id]

        structure = df['structure'].values[0]

        sequence = df['sequence'].values[0]

        bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

        my_plot_rna(bg, nt_id_num, df, max_value, min_value, nt=nt,

                    target=target_str, ax=ax[i//c, i%c])

def plot_low_target(n, target, target_str, nt=False, fig_size=None):

    r = (n // 5) if (n // 5 > 0) else 1

    c = 5

    

    if fig_size is None:

        fig_size = (40, 10*r)



    fig, ax = plt.subplots(r, c, figsize=fig_size)

    plt.suptitle(f'{r*c} smallest {target_str} values', fontsize=25)



    max_value = np.max(target[target_str])

    min_value = np.min(target[target_str])

    

    for i in range(r * c):

        nt_id = target.loc[len(target) - 1 - i].id

        nt_id_num = int(nt_id.split("_")[2])

        nt_id = nt_id[:12]

        df = train[train['id'] == nt_id]

        structure = df['structure'].values[0]

        sequence = df['sequence'].values[0]

        bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

        my_plot_rna(bg, nt_id_num, df, max_value, min_value, nt=nt,

                    target=target_str, ax=ax[i//c, i%c])
def plot_zero_target(n, target, target_str, nt=False, fig_size=None):

    target[target_str] = target[target_str].apply(abs)

    target = target.sort_values(by=[target_str], ascending=False)

    target = target.reset_index(drop=True)

    

    r = (n // 5) if (n // 5 > 0) else 1

    c = 5



    if fig_size is None:

        fig_size = (40, 10*r)

                 

    fig, ax = plt.subplots(r, c, figsize=fig_size)

    plt.suptitle(f'{r*c} {target_str} values closest to zero', fontsize=25)



    max_value = np.max(target[target_str])

    min_value = np.min(target[target_str])

    

    for i in range(r * c):

        nt_id = target.loc[len(target) - 1 - i].id

        nt_id_num = int(nt_id.split("_")[2])

        nt_id = nt_id[:12]

        df = train[train['id'] == nt_id]

        structure = df['structure'].values[0]

        sequence = df['sequence'].values[0]

        bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

        my_plot_rna(bg, nt_id_num, df, max_value, min_value, nt=nt,

                    target=target_str, ax=ax[i//c, i%c])
target_str = "reactivity"

target = make_target(target_str)

plot_distribution(target, target_str)
plot_high_target(15, target, target_str, nt=False)
plot_low_target(15, target, target_str, nt=False)
plot_zero_target(15, target, target_str, nt=False)
target_str = "deg_Mg_pH10"

target = make_target(target_str)

plot_distribution(target, target_str)
plot_high_target(15, target, target_str, nt=False)
plot_low_target(15, target, target_str)
plot_zero_target(15, target, target_str, nt=False)
target_str = "deg_Mg_50C"

target = make_target(target_str)

plot_distribution(target, target_str)
plot_high_target(15, target, target_str, nt=False)
plot_low_target(15, target, target_str, nt=False)
plot_zero_target(15, target, target_str, nt=False)
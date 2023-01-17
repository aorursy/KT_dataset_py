import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import OneHotEncoder



import matplotlib.pyplot as plt

import matplotlib.colors as mc

from matplotlib import cm

import seaborn as sns

import colorsys



import math



!conda install -y -c bioconda forgi

!conda install -y -c bioconda viennarna



import forgi.graph.bulge_graph as fgb

import forgi.visual.mplotlib as fvm

import forgi.threedee.utilities.vector as ftuv

import forgi



import RNA



import os



import warnings

warnings.filterwarnings('ignore')



PATH = "../input/stanford-covid-vaccine/"



train = pd.read_json(os.path.join(PATH, 'train.json'), lines=True)

test = pd.read_json(os.path.join(PATH, 'test.json'), lines=True)

submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
train = train[['id', 'sequence', 'structure', 'predicted_loop_type',

       'signal_to_noise', 'SN_filter', 'seq_length', 'seq_scored',

       'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',

       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',

       'deg_pH10', 'deg_Mg_50C', 'deg_50C']]



train['reactivity_min'] = train['reactivity'].apply(lambda x: np.min(x))

train['deg_Mg_pH10_min'] = train['deg_Mg_pH10'].apply(lambda x: np.min(x))

train['deg_pH10_min'] = train['deg_pH10'].apply(lambda x: np.min(x))

train['deg_Mg_50C_min'] = train['deg_Mg_50C'].apply(lambda x: np.min(x))

train['deg_50C_min'] = train['deg_50C'].apply(lambda x: np.min(x))



train['reactivity_max'] = train['reactivity'].apply(lambda x: np.max(x))

train['deg_Mg_pH10_max'] = train['deg_Mg_pH10'].apply(lambda x: np.max(x))

train['deg_pH10_max'] = train['deg_pH10'].apply(lambda x: np.max(x))

train['deg_Mg_50C_max'] = train['deg_Mg_50C'].apply(lambda x: np.max(x))

train['deg_50C_max'] = train['deg_50C'].apply(lambda x: np.max(x))



train['reactivity_mean'] = train['reactivity'].apply(lambda x: np.mean(x))

train['deg_Mg_pH10_mean'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))

train['deg_pH10_mean'] = train['deg_pH10'].apply(lambda x: np.mean(x))

train['deg_Mg_50C_mean'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))

train['deg_50C_mean'] = train['deg_50C'].apply(lambda x: np.mean(x))



train['reactivity_std'] = train['reactivity'].apply(lambda x: np.std(x))

train['deg_Mg_pH10_std'] = train['deg_Mg_pH10'].apply(lambda x: np.std(x))

train['deg_pH10_std'] = train['deg_pH10'].apply(lambda x: np.std(x))

train['deg_Mg_50C_std'] = train['deg_Mg_50C'].apply(lambda x: np.std(x))

train['deg_50C_std'] = train['deg_50C'].apply(lambda x: np.std(x))



test = test[['id', 'sequence', 'structure', 'predicted_loop_type', 'seq_length', 'seq_scored']]



all_data = train.append(test).reset_index(drop=True)



print(f"{len(train)} train data points + {len(test)} test data points = {len(all_data)} overall data point")

#all_data = all_data.reset_index()

print(f"\nSequences are fully unique ({all_data.sequence.nunique()}/{len(all_data)} uniques)")

print(f"Structures are NOT fully unique ({all_data.structure.nunique()}/{len(all_data)} uniques)")



temp = all_data.structure.value_counts()

temp = temp.to_frame().reset_index()

temp.columns = ['structure', 'counts']

temp = temp[temp.counts > 1]

print(f"\nThere are {len(temp)} reoccuring structures. Most re-occuring structures are duplicates or can re-occur up to 9 times in the dataset. \nLet's have a look at the most re-occuring structures:")



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))



for i in range(3):



    common_structure = temp.loc[i].structure

    """

    Edited from:

    * https://www.kaggle.com/erelin6613/openvaccine-rna-visualization/

    * https://www.kaggle.com/ricopue/second-structure-plot-and-info-with-forgi

    """

    sequence = all_data[all_data.structure == common_structure].iloc[0].sequence

    structure = all_data[all_data.structure == common_structure].iloc[0].structure

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None}, ax=ax[i])

    

    ax[i].set_title(f"Re-occurs {temp.loc[i].counts} times in dataset.")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))



idx = 0

sequence = all_data.iloc[idx].sequence

structure = all_data.iloc[idx].structure

bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]



fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None}, ax=ax[0])

ax[0].set_title("i0 is a Bulge (no opposite bulge), but i1 and i2 are internal loops", fontsize=16)



idx = 68

sequence = all_data.iloc[idx].sequence

structure = all_data.iloc[idx].structure

bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]



fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None}, ax=ax[1])

ax[1].set_title("m3 is an external loop, but m1 and m2 are multi loops", fontsize=16)

plt.show()

def get_bprna_features(df):

    

    df['E'] = 0

    df['S'] = 0

    df['H'] = 0

    df['B'] = 0

    df['I'] = 0

    df['M'] = 0

    df['X'] = 0

    prev_c = ''

    segment = ''

    segments = []

    for c in df.predicted_loop_type:

        if prev_c == c:

            segment = segment + c

        else:

            if 'E' in segment:

                df['E'] = df['E'] + 1

            elif 'S' in segment:

                df['S'] = df['S'] + 1

            elif 'H' in segment:

                df['H'] = df['H'] + 1

            elif 'B' in segment:

                df['B'] = df['B'] + 1

            elif 'I' in segment:

                df['I'] = df['I'] + 1

            elif 'M' in segment:

                df['M'] = df['M'] + 1

            elif 'X' in segment:

                df['X'] = df['X'] + 1

            segment = c

        prev_c = c

    if 'E' in segment:

        df['E'] = df['E'] + 1

    elif 'S' in segment:

        df['S'] = df['S'] + 1

    elif 'H' in segment:

        df['H'] = df['H'] + 1

    elif 'B' in segment:

        df['B'] = df['B'] + 1

    elif 'I' in segment:

        df['I'] = df['I'] + 1

    elif 'M' in segment:

        df['M'] = df['M'] + 1

    elif 'X' in segment:

        df['X'] = df['X'] + 1

        

    df['S'] = df['S'] / 2 # always has a matching partner

    df['I'] = df['I'] / 2 # always has a matching partner

    return df[['E', 'S', 'H', 'B', 'I', 'M', 'X']]



def get_forgi_features(df):

    sequence = df.sequence

    structure = df.structure

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    

    df['fiveprimes'] = len(list(bg.floop_iterator()))

    df['threeprimes'] = len(list(bg.tloop_iterator()))

    df['stems'] = len(list(bg.stem_iterator()))

    df['interior_loops'] = len(list(bg.iloop_iterator()))

    df['multiloops'] = len(list(bg.mloop_iterator()))

    df['hairpin_loops'] = len(list(bg.hloop_iterator()))

    

    return df[['fiveprimes', 'threeprimes', 'stems', 'interior_loops', 'multiloops', 'hairpin_loops']]



all_data[['E', 'S', 'H', 'B', 'I', 'M', 'X']] = all_data.apply(lambda x: get_bprna_features(x), axis=1)



all_data[['fiveprimes', 'threeprimes', 'stems', 'interior_loops', 'multiloops', 'hairpin_loops']] = all_data.apply(lambda x: get_forgi_features(x), axis=1)



print(f"Stems: {len(all_data[all_data.S != all_data.stems])} deviations between forgi and bpRNA.")

print(f"Hairpin Loops: {len(all_data[all_data.hairpin_loops != all_data.H])} deviations between forgi and bpRNA.")

print(f"Dangling End (Fiveprimes & Threeprimes): {len(all_data[all_data.E != (all_data.fiveprimes + all_data.threeprimes)])} deviations between forgi and bpRNA.")

print(f"Internal Loops and Bulges (Interior Loops): {len(all_data[all_data.interior_loops != (all_data.I + all_data.B)])} deviations between forgi and bpRNA.")

print(f"Multiloops and External Loops (Multiloop segment): {len(all_data[all_data.multiloops != (all_data.M + all_data.X)])} deviations between forgi and bpRNA.")

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))



sns.countplot(x=all_data.stems, ax=ax[0,0])

ax[0,0].set_title('Stems', fontsize=16)



sns.countplot(x=all_data.I, ax=ax[0,1])

ax[0,1].set_title('Internal Loops', fontsize=16)



sns.countplot(x=all_data.B, ax=ax[0,2])

ax[0,2].set_title('Bulges', fontsize=16)



sns.countplot(x=all_data.H, ax=ax[1,0])

ax[1,0].set_title('Hairpin Loops', fontsize=16)



sns.countplot(x=all_data.M, ax=ax[1,1])

ax[1,1].set_title('Multi Loop', fontsize=16)



sns.countplot(x=all_data.X, ax=ax[1,2])

ax[1,2].set_title('external loop', fontsize=16)



plt.show()

correlation_matrix = all_data[~all_data.reactivity_min.isna()][['stems', 'H', 'B', 'I', 'interior_loops', 'M', 'X', 'multiloops', 'E', 'fiveprimes']].corr()

matrix = np.triu(correlation_matrix)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))



ax[0].set_title('Correlation Matrix of Structure Segments', fontsize=16)



sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot = True, cmap='coolwarm', mask=matrix, ax=ax[0])



correlation_matrix = all_data[~all_data.reactivity_min.isna()][['stems', 'H', 'B', 'I', 'M', 'X', ]].corr()

matrix = np.triu(correlation_matrix)

ax[1].set_title('Condensed Correlation Matrix of Structure Segments', fontsize=16)



sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot = True, cmap='coolwarm', mask=matrix, ax=ax[1])

plt.show()
def custom_plot_rna(cg, coloring, ax=None):

    '''

    Edited from https://github.com/ViennaRNA/forgi/blob/master/forgi/visual/mplotlib.py

    '''

    RNA.cvar.rna_plot_type = 1

    coords = []

    bp_string = cg.to_dotbracket_string()

    if ax is None:

        ax = plt.gca()

    vrna_coords = RNA.get_xy_coordinates(bp_string)

    

    for i, _ in enumerate(bp_string):

        coord = (vrna_coords.get(i).X, vrna_coords.get(i).Y)

        coords.append(coord)

    coords = np.array(coords)

    

    # Now plot circles

    for i, coord in enumerate(coords):

        if i < len(coloring):

            c = cm.coolwarm(coloring[i])

        else: 

            c = 'grey'

        h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))

        c=colorsys.hls_to_rgb(h,l,s)

        circle = plt.Circle((coord[0], coord[1]),color=c)

        ax.add_artist(circle)



    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),

                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),

               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),

                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))



    width = datalim[1][0] - datalim[0][0]

    height = datalim[1][1] - datalim[0][1]



    ax.set_aspect('equal', 'datalim')

    ax.update_datalim(datalim)

    ax.autoscale_view()

    ax.set_axis_off()



    return (ax, coords)



def plot_structure_with_target_var(idx):

    sequence = all_data.iloc[idx].sequence

    structure = all_data.iloc[idx].structure



    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))



    coloring = all_data.iloc[idx].reactivity

    coloring = [min(max(((c-(-np.percentile(train.reactivity_max, 90)))/(np.percentile(train.reactivity_max, 90)-(-np.percentile(train.reactivity_max, 90)))), 0),1) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[0])

    ax[0].set_title('reactivity', fontsize=16)



    coloring = all_data.iloc[idx].deg_Mg_pH10

    coloring = [min(max(((c-(-np.percentile(train.deg_Mg_pH10_max, 90)))/(np.percentile(train.deg_Mg_pH10_max, 90)-(-np.percentile(train.deg_Mg_pH10_max, 90)))), 0),1) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[1])

    ax[1].set_title('deg_Mg_pH10', fontsize=16)



    coloring = all_data.iloc[idx].deg_pH10

    coloring = [min(max(((c-(-np.percentile(train.deg_pH10_max, 90)))/(np.percentile(train.deg_pH10_max, 90)-(-np.percentile(train.deg_pH10_max, 90)))), 0),1) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[2])

    ax[2].set_title('deg_pH10', fontsize=16)



    coloring = all_data.iloc[idx].deg_Mg_50C

    coloring = [min(max(((c-(-np.percentile(train.deg_Mg_50C_max, 90)))/(np.percentile(train.deg_Mg_50C_max, 90)-(-np.percentile(train.deg_Mg_50C_max, 90)))), 0),1) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[3])

    ax[3].set_title('deg_Mg_50C', fontsize=16)



    coloring = all_data.iloc[idx].deg_50C

    coloring = [min(max(((c-(-np.percentile(train.deg_50C_max, 90)))/(np.percentile(train.deg_50C_max, 90)-(-np.percentile(train.deg_50C_max, 90)))), 0),1) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[4])

    ax[4].set_title('deg_50C', fontsize=16)



    plt.show()



for i in range(3):

    plot_structure_with_target_var(i)
train_seqpos = pd.DataFrame(columns=[ 'id_seqpos', 'predicted_loop_type', 'sequence', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'])



for i in range(len(train)):

    df = train.loc[i]

    new_df = pd.DataFrame(data={'id': df.id, 'pos': list(range(df.seq_scored)), 

                                'predicted_loop_type': list(df.predicted_loop_type)[:(df.seq_scored)],

                                'sequence': list(df.sequence)[:(df.seq_scored)],

                                'reactivity': list(df.reactivity)[:(df.seq_scored)], 

                                'deg_Mg_pH10': list(df.deg_Mg_pH10)[:(df.seq_scored)],

                                'deg_pH10': list(df.deg_pH10)[:(df.seq_scored)],

                                'deg_Mg_50C': list(df.deg_Mg_50C)[:(df.seq_scored)],

                                'deg_50C': list(df.deg_50C)[:(df.seq_scored)]})



    new_df['id_seqpos'] = new_df.apply(lambda x: f"{x['id']}_{x['pos']}", axis=1)

    new_df = new_df.drop(['id', 'pos'], axis=1)

    new_df = new_df[[ 'id_seqpos', 'predicted_loop_type', 'sequence', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']] 

    train_seqpos = train_seqpos.append(new_df)



train_seqpos = train_seqpos.reset_index(drop=True)



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_seqpos[['predicted_loop_type', 'sequence']]))



OH_cols_train.columns = OH_encoder.get_feature_names(['predicted_loop_type', 'sequence'])



OH_cols_train.index = train_seqpos.index



train_seqpos = pd.concat([train_seqpos, OH_cols_train], axis=1)



fig, ax = plt.subplots(figsize=(12, 10))



correlation_matrix = train_seqpos.corr()

matrix = np.triu(correlation_matrix)



sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot = True, cmap='coolwarm', mask=matrix)

plt.show()



train_seqpos.head()
outliers = train[((train.reactivity_max == 0) & (train.reactivity_min == 0)) 

                 | ((train.deg_Mg_50C_max == 0) & (train.deg_Mg_50C_min == 0))

                 | ((train.deg_Mg_pH10_max == 0) & (train.deg_Mg_pH10_min == 0))

                | ((train.deg_pH10_max == 0) & (train.deg_pH10_min == 0))

                | ((train.deg_50C_max == 0) & (train.deg_50C_min == 0))]

print(f"There are {len(outliers)} data points where at least one of reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, or deg_50C is fully zero.")

print(outliers.id)



for i in outliers.index:

    plot_structure_with_target_var(i)
# Garbage collection of code for later usage

"""





fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 3))

fig.suptitle('Distribution of Min. Values', fontsize=16, y=1.2)



sns.kdeplot(data = train.reactivity_min, ax= ax[0])

ax[0].set_title('reactivity')



sns.kdeplot(data = train.deg_Mg_pH10_min, ax= ax[1])

ax[1].set_title('deg_Mg_pH10')



sns.kdeplot(data = train.deg_pH10_min, ax= ax[2])

ax[2].set_title('deg_pH10')



sns.kdeplot(data = train.deg_Mg_50C_min, ax= ax[3])

ax[3].set_title('deg_Mg_50C')



sns.kdeplot(data = train.deg_50C_min, ax= ax[4])

ax[4].set_title('deg_50C')



plt.show()



fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 3))

fig.suptitle('Distribution of Max. Values', fontsize=16, y=1.2)



sns.kdeplot(data = train.reactivity_max, ax= ax[0])

ax[0].set_title('reactivity')



sns.kdeplot(data = train.deg_Mg_pH10_max, ax= ax[1])

ax[1].set_title('deg_Mg_pH10')



sns.kdeplot(data = train.deg_pH10_max, ax= ax[2])

ax[2].set_title('deg_pH10')



sns.kdeplot(data = train.deg_Mg_50C_max, ax= ax[3])

ax[3].set_title('deg_Mg_50C')



sns.kdeplot(data = train.deg_50C_max, ax= ax[4])

ax[4].set_title('deg_50C')



plt.show()



'''fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

fig.suptitle('Count Distribution according to seq_scored (68 or 91)', fontsize=16)

sns.kdeplot(data=all_data[all_data.seq_scored == 68].stems, ax=ax[0,0])

sns.kdeplot(data=all_data[all_data.seq_scored == 91].stems, ax=ax[0,0])

ax[0,0].set_title('Stems', fontsize=16)

ax[0,0].legend(title='Seq_scored', loc='upper left', labels=['68', '91'])



sns.kdeplot(data=all_data[all_data.seq_scored == 68].interior_loops, ax=ax[0,1])

sns.kdeplot(data=all_data[all_data.seq_scored == 91].interior_loops, ax=ax[0,1])

ax[0,1].set_title('Interior Loops', fontsize=16)

ax[0,1].legend(title='Seq_scored', loc='upper left', labels=['68', '91'])



sns.kdeplot(data=all_data[all_data.seq_scored == 68].multiloops, ax=ax[1,0])

sns.kdeplot(data=all_data[all_data.seq_scored == 91].multiloops, ax=ax[1,0])

ax[1,0].set_title('Multiloops', fontsize=16)

ax[1,0].legend(title='Seq_scored', loc='upper left', labels=['68', '91'])



sns.kdeplot(data=all_data[all_data.seq_scored == 68].hairpin_loops, ax=ax[1,1])

sns.kdeplot(data=all_data[all_data.seq_scored == 91].hairpin_loops, ax=ax[1,1])

ax[1,1].set_title('Hairpin Loops', fontsize=16)

ax[1,1].legend(title='Seq_scored', loc='upper left', labels=['68', '91'])

plt.show()'''



temp_df = train[(train.reactivity_max > 20) & (train.reactivity_min < -20)]



print(f"There are {len(temp_df)} training datapoints with an absolute min./max. reactivity over 20.")

print(f"Within these {len(temp_df)}, there are {temp_df.structure.nunique()} unique structures.")



fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))



temp = temp_df.structure.value_counts()

temp = temp.to_frame().reset_index()

temp.columns = ['structure', 'counts']



for i in range(temp_df.structure.nunique()):

    col =  i % 4

    row = i // 4

    

    common_structure = temp.loc[i].structure



    sequence = train[train.structure == common_structure].iloc[0].sequence

    structure = train[train.structure == common_structure].iloc[0].structure

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None}, ax=ax[row, col])

    

    ax[row, col].set_title(f"Re-occurs {temp.loc[i].counts} times in dataset.")

    

    

    

    

    #####





def plot_structure_with_forgi(idx):



    plt.figure(figsize=(8,8))

    sequence = train.loc[idx, 'sequence']

    structure = train.loc[idx, 'structure']

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    fvm.plot_rna(bg, lighten=0.5)

    plt.show()

    

def convert_to_grouped_structure(structure_string):

    grouped_structure = np.zeros(len(structure_string))

    for idx, char in enumerate(structure_string):

        if char == '.':

            if idx != 0:

                grouped_structure[idx] = grouped_structure[idx-1]

            else:

                grouped_structure[idx] = 0

        elif char == "(":

            grouped_structure[idx] = grouped_structure[idx-1] + 1

        else:

            grouped_structure[idx] = grouped_structure[idx-1] - 1

    return grouped_structure



def plot_structures(idx):

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 3))

    df = pd.DataFrame({'structure' : convert_to_grouped_structure(train.structure.loc[idx])[:68], 

                       'reactivity': train.reactivity.loc[idx], 

                       'deg_Mg_pH10': train.deg_Mg_pH10.loc[idx],

                       'deg_pH10': train.deg_pH10.loc[idx],

                       'deg_Mg_50C': train.deg_Mg_50C.loc[idx],

                       'deg_50C': train.deg_50C.loc[idx]})

    sns.scatterplot(y=df['structure'], 

                    x=df.index, 

                    hue=df.reactivity, 

                    marker='o', 

                    s=20,

                    palette = 'cool', 

                    hue_norm = matplotlib.colors.Normalize(vmin=np.percentile(train['reactivity_min'], 10), vmax=np.percentile(train['reactivity_max'], 90), clip=True), 

                    ax= ax[0])

    ax[0].legend([],[], frameon=False)

    ax[0].set_title('reactivity')



    sns.scatterplot(y=df['structure'], 

                    x=df.index, 

                    hue=df.deg_Mg_pH10, 

                    marker='o', 

                    s=20,

                    palette = 'cool', 

                    hue_norm = matplotlib.colors.Normalize(vmin=np.percentile(train['deg_Mg_pH10_min'], 10), vmax=np.percentile(train['deg_Mg_pH10_max'], 90), clip=True), 

                    ax= ax[1])

    ax[1].legend([],[], frameon=False)

    ax[1].set_title('deg_Mg_pH10')



    sns.scatterplot(y=df['structure'], 

                    x=df.index, 

                    hue=df.deg_pH10, 

                    marker='o', 

                    s=20,

                    palette = 'cool', 

                    hue_norm = matplotlib.colors.Normalize(vmin=np.percentile(train['deg_pH10_min'], 10), vmax=np.percentile(train['deg_pH10_max'], 90), clip=True), 

                    ax= ax[2])

    ax[2].legend([],[], frameon=False)

    ax[2].set_title('deg_pH10')



    sns.scatterplot(y=df['structure'], 

                    x=df.index, 

                    hue=df.deg_Mg_50C, 

                    marker='o', 

                    s=20,

                    palette = 'cool', 

                    hue_norm = matplotlib.colors.Normalize(vmin=np.percentile(train['deg_Mg_50C_min'], 10), vmax=np.percentile(train['deg_Mg_50C_max'], 90), clip=True), 

                    ax= ax[3])

    ax[3].legend([],[], frameon=False)

    ax[3].set_title('deg_Mg_50C')



    sns.scatterplot(y=df['structure'], 

                    x=df.index, 

                    hue=df.deg_50C, 

                    marker='o', 

                    s=20,

                    palette = 'cool', 

                    hue_norm = matplotlib.colors.Normalize(vmin=np.percentile(train['deg_50C_min'], 10), vmax=np.percentile(train['deg_50C_max'], 90), clip=True), 

                    ax= ax[4])

    ax[4].legend([],[], frameon=False)

    ax[4].set_title('deg_50C')

    plt.suptitle(f"{train.structure.loc[idx][:68]}\n{train.sequence.loc[idx][:68]}", fontsize=16, y=1.2)



    plt.show()

    

for i in range(5):

    plot_structures(i)

    plot_structure_with_forgi(i)

    

outliers = train[(train.reactivity_max > 20) | (train.reactivity_min < -20) 

                 | (train.deg_Mg_pH10_max < -10) | (train.deg_Mg_pH10_min < -10)

                 | (train.deg_pH10_max < -40) | (train.deg_pH10_min < -40)

                 | (train.deg_Mg_50C_max < -20) | (train.deg_Mg_50C_min < -20) 

                 | (train.deg_50C_max < -40) | (train.deg_50C_min < -40)]



outliers[['reactivity_max', 'reactivity_min', 'deg_Mg_pH10_max', 'deg_Mg_pH10_min', 'deg_pH10_max', 'deg_pH10_min', 'deg_Mg_50C_max', 'deg_Mg_50C_min', 'deg_50C_max', 'deg_50C_min']]





all_data['seq_length_check'] = all_data['sequence'].apply(lambda x: len(x))

print(f"seq_length has {len(all_data[all_data.seq_length != all_data.seq_length_check][['sequence', 'seq_length', 'seq_length_check']])} implausible entries.")



def check_seq_scored_for_plausibility(feature):

    train['check'] = train[feature].apply(lambda x: len(x))

    print(f"{feature} has {len(train[train.check != train.seq_scored][['check', 'seq_scored']])} implausible entries for seq_scored.")



check_seq_scored_for_plausibility('reactivity')

check_seq_scored_for_plausibility('reactivity_error')



check_seq_scored_for_plausibility('deg_pH10')

check_seq_scored_for_plausibility('deg_error_pH10')



check_seq_scored_for_plausibility('deg_Mg_pH10')

check_seq_scored_for_plausibility('deg_error_Mg_pH10')



check_seq_scored_for_plausibility('deg_50C')

check_seq_scored_for_plausibility('deg_error_50C')



check_seq_scored_for_plausibility('deg_Mg_50C')

check_seq_scored_for_plausibility('deg_error_Mg_50C')







im = np.load(os.path.join(PATH, "bpps/id_00073f8be.npy"))



def check_seq_scored_for_plausibility(feature):

    test['check'] = test[feature].apply(lambda x: len(x))

    print(f"{feature} has {len(test[test.check != test.seq_scored][['check', 'seq_scored']])} implausible entries for seq_scored.")



check_seq_scored_for_plausibility('reactivity')

check_seq_scored_for_plausibility('deg_Mg_pH10')

check_seq_scored_for_plausibility('deg_pH10')

check_seq_scored_for_plausibility('deg_Mg_50C')

check_seq_scored_for_plausibility('deg_50C')"""
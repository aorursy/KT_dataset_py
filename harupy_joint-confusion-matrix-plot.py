import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
np.random.seed(42)



cm = np.random.randint(0, 50, size=(4, 4))

cm
class JointConfusionMatrix:

    """

    Ref. https://github.com/mwaskom/seaborn/blob/master/seaborn/axisgrid.py#L1551

    """

    def __init__(self, cm, height=6, ratio=5, space=.2,

                 dropna=True, xlim=None, ylim=None, size=None):

        

        # Set up the subplot grid

        f = plt.figure(figsize=(height, height))

        gs = plt.GridSpec(ratio + 1, ratio + 1)



        ax_joint = f.add_subplot(gs[1:, :-1])

        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)

        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)



        self.fig = f

        self.ax_joint = ax_joint

        self.ax_marg_x = ax_marg_x

        self.ax_marg_y = ax_marg_y

        self.cm = cm



        # Turn off tick visibility for the measure axis on the marginal plots

        plt.setp(ax_marg_x.get_xticklabels(), visible=False)

        plt.setp(ax_marg_y.get_yticklabels(), visible=False)



        # Turn off the ticks on the density axis for the marginal plots

        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)

        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)

        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)

        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)

        plt.setp(ax_marg_x.get_yticklabels(), visible=False)

        plt.setp(ax_marg_y.get_xticklabels(), visible=False)

        

        ax_marg_x.yaxis.grid(False)

        ax_marg_y.xaxis.grid(False)



        if xlim is not None:

            ax_joint.set_xlim(xlim)

        if ylim is not None:

            ax_joint.set_ylim(ylim)



        # Make the grid look nice

        sns.utils.despine(f)

        sns.utils.despine(ax=ax_marg_x, left=True)

        sns.utils.despine(ax=ax_marg_y, bottom=True)

        f.tight_layout()

        f.subplots_adjust(hspace=space, wspace=space)

        

        

    def make_annotation(self, cm, cm_norm, normalize=True):

        annot = []

        nrows, ncols = cm.shape

        base = '{}\n({:.2f})'

        for ir in range(nrows):

            annot.append([])

            for ic in range(ncols):

                annot[ir].append(base.format(cm[ir, ic], cm_norm[ir, ic]))

       

        return np.array(annot)

            

    def plot(self, labels, normalize=True):

        true_dist = cm.sum(axis=1)

        pred_dist = cm.sum(axis=0)

        pos = np.arange(cm.shape[0]) + 0.5

        

        cm_norm = cm / true_dist.reshape(-1, 1)

        annot = self.make_annotation(cm, cm_norm)

        

        FONTSIZE = 20

        

        # plot confusion matrix as heatmap

        sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,

                    annot=annot, fmt='s', annot_kws={'fontsize': FONTSIZE},

                    linewidths=0.2, cbar=False, square=True, ax=self.ax_joint)

        self.ax_joint.set_xlabel('Predicted label', fontsize=FONTSIZE)

        self.ax_joint.set_xticklabels(labels, fontsize=FONTSIZE)

        

        self.ax_joint.set_ylabel('True label', fontsize=FONTSIZE)

        self.ax_joint.set_yticklabels(labels, fontsize=FONTSIZE)

        

        props = {'align': 'center'}

        

        # plot predicted label distribution

        self.ax_marg_x.bar(pos, pred_dist / pred_dist.sum(), **props)

        self.ax_marg_x.set_title('Predicted label distribution', fontsize=FONTSIZE)

        

        # plot true label distribution

        self.ax_marg_y.barh(pos, true_dist / true_dist.sum(), **props)

        self.ax_marg_y.text(0.5, 1.5, 'True label distribution',

                            rotation=270, fontsize=FONTSIZE)
g = JointConfusionMatrix(cm, height=10)

g.plot(labels=['a', 'b', 'c', 'd'])
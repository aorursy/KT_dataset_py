%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import pickle

# from graphutils import line3
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



def histgram_act_dist(dists):

    plt.style.use("default")

    layer_num = len(dists.keys())

    fig = plt.figure(figsize=(20,2))

    fig.suptitle("Activation Distributions", y=1.15, fontsize=18)

    ax = fig.subplots(1, layer_num)

    for i, dist in enumerate(dists):

        key = "layer" + str(i)

        ax[i].hist(dists[key], bins=5)

        ax[i].set_title(key)

        ax[i].set_xlim(0.0, 2.0)

        ax[i].set_xlabel("Value after Affine-Transformation")

        if i == 0:

            ax[i].set_ylabel("Count")

        ax[i].set_ylim(0, 12)

    plt.show()



def bar_init_weight_effect(df):

    plt.style.use("default")

    df = df.drop(["init_weight", "v_avg_accuracy", "v_argmax", "memo"], axis=1)

    ax = df.plot.bar(x="act_func", y=["v_avg_loss", "v_max_accuracy"], figsize=(8,3))

    ax.set_title("Effect of Initial Weight Change")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    arrwprp_down = dict(arrowstyle="<-", color="deepskyblue", shrinkA=5, shrinkB=5, patchA=None, patchB=None, connectionstyle="arc3,rad=0.1")

    arrwprp_up = dict(arrowstyle="<-", color="deeppink", shrinkA=5, shrinkB=5, patchA=None, patchB=None, connectionstyle="arc3,rad=-0.1")

    for i in range(3):

        pos1 = 2*i

        pos2 = 2*i+1

        ax.annotate("Down!", color="deepskyblue",

            xy=(pos1-0.15, df.v_avg_loss[pos1]), xycoords='data',

            xytext=(pos2-0.3, df.v_avg_loss[pos2]), textcoords='data',

            arrowprops=arrwprp_down

        )

        ax.annotate("Up!",color="deeppink",

            xy=(pos1+0.1, df.v_max_accuracy[pos1]-0.01), xycoords='data',

            xytext=(pos2+0.05, df.v_max_accuracy[pos2]+0.01), textcoords='data',

            arrowprops=arrwprp_up

        )



    plt.show()



def plot_BN_effect(df_dict):

    plt.style.use("default")

    fig = plt.figure(figsize=(11, 3.5))

    fig.suptitle("Effect of Batch Normalization", y=1.05, fontsize=18)

    ax = fig.subplots(1, len(df_dict))

    for i, (key, val) in enumerate(df_dict.items()):

        ax1 = ax[i]



        accuracy_color = "darkorange"

        ax1.plot(val.epoch, val.l_accuracy, label="l_accu: accuracy on learning data", linestyle="dashed", color=accuracy_color)

        ax1.plot(val.epoch, val.v_accuracy, label="v_accu: accuracy on validation data", linestyle="solid", color=accuracy_color)

        ax1.set_xlabel("epoch")

        ax1.set_xlim(-3, 103)

        ax1.set_ylabel("accuracy", color=accuracy_color, horizontalalignment="left")

        ax1.set_ylim(0.84, 1.03)

        ax1.set_yticks([0.95,0.975,1.0])

        ax1.tick_params(axis="y", colors=accuracy_color)

        ax1.set_title(key)

        if i == len(df_dict)-1:

            ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

        ax1.grid(which='major',color='lightgray',linestyle='--')

        ax1.grid(which='minor',color='lightgray',linestyle='--')

        epochidx = 16

        ax1.annotate("l_accu",

                     xy=(val.epoch[epochidx], val.l_accuracy[epochidx]),

                     xytext=(val.epoch[epochidx], val.l_accuracy[epochidx]+0.014),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color="olive"),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=accuracy_color, alpha=0.1)

                    )

        ax1.annotate("v_accu",

                     xy=(val.epoch[epochidx], val.v_accuracy[epochidx]),

                     xytext=(val.epoch[epochidx], val.v_accuracy[epochidx]-0.021),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color="olive"),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=accuracy_color, alpha=0.1)

                    )



        ax2 = ax1.twinx()

        loss_color = "royalblue"

        ax2.plot(val.epoch, val.l_loss, label="l_loss: loss on learning data", linestyle="dashed", color=loss_color)

        ax2.plot(val.epoch, val.v_loss, label="v_loss: loss on validation data", linestyle="solid", color=loss_color)

        ax2.set_ylabel("loss", color=loss_color, horizontalalignment="right")

        ax2.set_ylim(-0.08, 0.5)

        ax2.set_yticks([0.0,0.05,0.1,0.15])

        ax2.tick_params(axis="y", colors=loss_color)

        if i == len(df_dict)-1:

            ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 0.8))

        ax2.grid(which='major',color='lightgray',linestyle='--')

        ax2.grid(which='minor',color='lightgray',linestyle='--')

        epochidx = 78

        ax2.annotate("l_loss",

                     xy=(val.epoch[epochidx], val.l_loss[epochidx]),

                     xytext=(val.epoch[epochidx], val.l_loss[epochidx]-0.06),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=loss_color),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=loss_color, alpha=0.1)

                    )

        ax2.annotate("v_loss",

                     xy=(val.epoch[epochidx], val.v_loss[epochidx]),

                     xytext=(val.epoch[epochidx], val.v_loss[epochidx]+0.04),

                     fontsize=9, 

                     arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=loss_color),

                     bbox=dict(boxstyle="round4, pad=0.2", fc=loss_color, alpha=0.1)

                    )



    plt.subplots_adjust(wspace=1, hspace=0)

    plt.tight_layout()

    plt.show()

    

def line(df, title="", cols=["l_loss", "l_accuracy", "v_loss", "v_accuracy"]):

    df.plot.line(x="epoch", y=cols)

#     df.plot.line(x="epoch", y=["l_loss", "l_accuracy", "v_loss", "v_accuracy"])

    plt.title(title)

    plt.show()



def line2(df, title="", cols=["l_loss", "l_accuracy", "v_loss", "v_accuracy"]):

    fig = plt.figure(figsize=(12,3))

    ax = fig.subplots(1,2)

    

    ax[0].plot(df.epoch, df.l_accuracy, label="l_accuracy", linestyle="dashed", color="darkorange")

    ax[0].plot(df.epoch, df.v_accuracy, label="v_accuracy", linestyle="solid", color="darkorange")

    ax[0].legend(loc='upper right', bbox_to_anchor=(1.4,1.0))

    ax[0].set_xlim(-3, 103)

    ax[0].set_ylim(0.85, 1.02)

    ax[0].grid(which='major',color='lightgray',linestyle='--')

    

    ax[1].plot(df.epoch, df.l_loss, label="l_loss", linestyle="dashed", color="royalblue")

    ax[1].plot(df.epoch, df.v_loss, label="v_loss", linestyle="solid", color="royalblue")

    ax[1].legend(loc='upper right', bbox_to_anchor=(1.3,1.0))

    ax[1].set_xlim(-3, 103)

    ax[1].set_ylim(-0.05, 0.55)

    ax[1].grid(which='major',color='lightgray',linestyle='--')



    plt.subplots_adjust(wspace=0.5, hspace=0)

    plt.title(title)

    plt.show()

    

def line3(title="", cols=["l_loss", "l_accuracy", "v_loss", "v_accuracy"]):

    path = "../input/dl-workshop-homework2-csv/dl_workshop_homework2_csv/" + title + "/perform.csv"

    #path = "./" + title + "/perform.csv"

    df = pd.read_csv(path)



    fig = plt.figure(figsize=(4.5, 3))

    ax1 = fig.subplots(1,1)

    

    ax1.plot(df.epoch, df.l_accuracy, label="l_accuracy", linestyle="dashed", color="darkorange")

    ax1.plot(df.epoch, df.v_accuracy, label="v_accuracy", linestyle="solid", color="darkorange")

    ax1.set_xlim(-3, 103)

    ax1.set_ylim(0.85, 1.02)

    ax1.set_yticks([0.95,0.975,1.0])

    ax1.grid(which='major',color='lightgray',linestyle='--')

    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    

    ax2 = ax1.twinx()

    ax2.plot(df.epoch, df.l_loss, label="l_loss", linestyle="dashed", color="royalblue")

    ax2.plot(df.epoch, df.v_loss, label="v_loss", linestyle="solid", color="royalblue")

    ax2.set_xlim(-3, 103)

    ax2.set_ylim(-0.05, 0.55)

    ax2.set_yticks([0.0,0.05,0.1,0.15])

    ax2.grid(which='major',color='lightgray',linestyle='--')

    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 0.75))



    plt.subplots_adjust(wspace=0.5, hspace=0)

    plt.title(title)

    plt.show()

# No.1:基本モデル：Conv2x2x4_Pool2x2_Affine100x1_Affine5x1

line3(title="CNN_Minibatch-ReLU-AdaGrad_Conv2x2x4-Pool2x2_Affine100x1_Affine5x1")
# No.2:基本モデルから畳み込み層のフィルターサイズを3に増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv3x3x4-Pool2x2_Affine100x1_Affine5x1")
# No.3:基本モデルから畳み込み層のフィルターサイズを5に増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv5x5x4-Pool2x2_Affine100x1_Affine5x1")
# No.4:基本モデルから畳み込み層のチャネル数を8に増やした。

line3(title="CNN_Minibatch-ReLU-AdaGrad_Conv2x2x8-Pool2x2_Affine100x1_Affine5x1")
# No.5:基本モデルから畳み込み層のチャネル数を16に増やした。

line3(title="CNN_Minibatch-ReLU-AdaGrad_Conv2x2x16-Pool2x2_Affine100x1_Affine5x1")
# No.6:基本モデルから畳み込み層のフィルターチャンネル数を32に増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv2x2x32-Pool2x2_Affine100x1_Affine5x1")
# No.7:基本モデルから畳み込み層のフィルターサイズとチャネル数を両方増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv3x3x32-Pool2x2_Affine100x1_Affine5x1")
# No.8:基本モデルからプーリング層のフィルターサイズを増やした。※ただし、プーリング層への入力画像サイズである27x27を割り切るように同時にストライドを3に増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv2x2x4-Pool3x3str3_Affine100x1_Affine5x1")
# No.9:基本モデルから畳み込み層のフィルターサイズを3にした状態で、プーリング層のフィルターサイズを5に増やした。

line3("CNN_Minibatch-ReLU-AdaGrad_Conv3x3x4-Pool5x5_Affine100x1_Affine5x1")
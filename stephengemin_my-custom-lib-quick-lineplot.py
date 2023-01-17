#lineplot.py

'''DOCSTRING:

I found that I was typing too many lines of code each time I wanted to create a simple plot

Plus, I could never remember the exact formatting I used when creating a new plot



I wrote this to reduce the number of lines of code to a max of 3 to generate a plot



Restrictions/Notes:

1. Only works with one y-axis at this point

2. use None for xmin/xmax/ymin/ymax if you want autoscaling

    xmin/max/ymin/ymax: int

3. num_minor_ticks:     int

4. save_loc:            str (if saving outside of relative path, use absolute path)

5. plot_name:           str

'''

import matplotlib

import matplotlib.pyplot as plt

import seaborn



class LinePlot():

    

    def __init__(self, xmin, xmax, ymin, ymax, minor_ticks=False, num_minor_ticks=0, 

                 xaxis_label=None, yaxis_label=None, title=None, 

                 save_plot=False, save_loc=None, plot_name=None):

        self.xmin = xmin

        self.xmax = xmax

        self.ymin = ymin

        self.ymax = ymax

        self.minor_ticks = minor_ticks

        self.num_minor_ticks = num_minor_ticks

        self.xaxis_label = xaxis_label

        self.yaxis_label = yaxis_label

        self.title = title

        self.save_plot = save_plot

        self.save_loc = save_loc

        self.plot_name = plot_name

    

    def set_style(self, styles):

        '''DOCSTRING:

        Use the command print(plt.style.available) to see list of all pre-built styles available

        '''

        plt.style.use(styles)

        matplotlib.rc('font', family="Sans-Serif", size=16)

    

    def add_legend(self, ax):

        h, l = ax.get_legend_handles_labels()

        ax.legend(h, l, loc='best', shadow=True, fontsize=14, title='Legend', frameon=True, fancybox=True, facecolor='white');

        

    def set_axes_labels(self, ax):

        ax.set_xlabel(self.xaxis_label)

        ax.set_ylabel(self.yaxis_label)

    

    def set_axes_lims(self, ax):

        ax.set_ylim(self.ymin,self.ymax)

        ax.set_xlim(self.xmin,self.xmax)

    

    def set_miniticks(self, ax):

        if self.minor_ticks:

            ax.minorticks_on()

            ax.grid(which='minor', linestyle=':')

            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(self.num_minor_ticks))

    

    def add_to_plot(self,x, y, linestyle, series_labels, ax):

        color_map = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'black']

        color_counter = 0

        if len(y) == len(series_labels):

            for chn in range(len(y)):

                if color_counter > len(color_map):

                    color_counter = 0

                ax.plot(x[chn], y[chn], linestyle=linestyle, color=color_map[chn], label=series_labels[chn])

                color_counter += 1

    

    def create_plot(self, x, y, series_labels):

        '''DOCSTRING:

        If plotting multiple channels, all inputs must be in list format 

        and they must have the same length!!!

        '''

        fig, ax = plt.subplots()

        fig.set_size_inches(14,10)

        self.add_to_plot(x, y, '-', series_labels, ax)

        ax.set_title(self.title, pad=10)

        self.set_axes_lims(ax)

        self.set_axes_labels(ax)

        self.add_legend(ax)

        self.set_miniticks(ax)

        fig.tight_layout(pad=1.5)

        if self.save_plot:

            fig.savefig(os.path.join(self.save_loc + self.plot_name + '.pdf'))
# get list of all pre-built styles from matplotlib

plt.style.available
# Example code using the attached dataset and the coding above

# normally I call this in as a module from my custom library

import numpy as np

import pandas as pd

df = pd.read_csv("../input/Motored Engine Data.csv", delimiter="\t")



#example with only 1 thing to plot

_plot = LinePlot(0, 120,-1000, 1000, xaxis_label = 'Time [s]', yaxis_label = 'Strain [µm]', title='Strain vs Time')

_plot.set_style(['seaborn'])

_plot.create_plot([df.Time], [df.D_Int_Spg_653mm_ue], ['D_Int_Spg_653mm_ue']) 
#example with multiple things to plot

# I'm invoking the option to include minigrid

_plot = LinePlot(0, 120,-1000, 1000, minor_ticks=True, num_minor_ticks=5,

                 xaxis_label = 'Time [s]', yaxis_label = 'Strain [µm]', title='Strain vs Time')

_plot.set_style(['seaborn-poster'])

_plot.create_plot([df.Time, df.Time, df.Time], 

                  [df.D_Int_Spg_653mm_ue, df.B_Int_Spg_80mm_ue, df.C_Int_Spg_OutterBolt_ue], 

                  ['D_Int_Spg_653mm_ue', 'B_Int_Spg_80mm_ue', 'C_Int_Spg_OutterBolt_ue']) 
# fig, ax = plt.subplots()

# fig.set_size_inches(10,6)

# for j in range(0,temp_df.shape[1],4):

#     ax.plot(temp_df.iloc[:,j+3], temp_df.iloc[:,j], 'blue', linewidth=1, label=stats_summary.loc[counter, 'Specimen Identification'] + '_Corrected')

#     ax.plot(temp_df.iloc[:,j+1], temp_df.iloc[:,j], 'green', linewidth=1, label=stats_summary.loc[counter, 'Specimen Identification'])

#     counter += 1



# # Add rotor and load cell FEA results

# ax.plot(FEA_df.loc[:,'Part Displacement'], FEA_df.loc[:,'Part Force'], 'r-', linewidth=1, label='Part FEA')

# ax.plot(FEA_df.loc[:,'LCell Displacement'], FEA_df.loc[:,'LCell Force'], '-', color='orange', linewidth=1, label='Load Cell FEA')    

# h, l = ax.get_legend_handles_labels()

# ax.legend(h, l, loc='lower right', shadow=True)



# ax.set_ylabel('Force [N]', fontsize=12)

# ax.tick_params(axis='y', labelsize=12)

# ax.set_ylim(0,50000)

# ax.set_xlabel('Position [mm]', fontsize=12)

# ax.set_xlim(0,0.35)

# ax.set_title('Force vs. Corrected Position', fontsize=16)

# ax.grid(which='major', linestyle='-')

# ax.minorticks_on()

# ax.grid(which='minor', linestyle=':')



# plt.tight_layout(pad=1.5)

# plt.savefig(user_desktop_temp + 'Summary.pdf')
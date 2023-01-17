from IPython.display import HTML

%matplotlib inline
!apt-get -y install ffmpeg
from scipy.stats import beta

import matplotlib.pyplot as plt

import numpy as np



# helper function for plotting

def plot_beta(a,b,ax, print_interval=True):

    ax.set_xlabel("p")

    ax.set_ylabel("probability density")

    x = np.linspace(0.00,1, 100)

    label = "$\\alpha= " + str(a) + ", \\beta=" + str(b) + "$"

    dist = beta(a,b)

    # plot density

    ax.plot(x, dist.pdf(x),

            lw=2, alpha=0.6, label=label)

    # determine the 95% HDI

    if print_interval:

        print("Interval containing 95% of the distribution: ", dist.interval(0.95))
fig, ax = plt.subplots(1,1)

plot_beta(10,10,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(100,100,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(1000,1000,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(10,1,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(1,1,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
# the animations work for me on Ubuntu

from scipy.stats import beta, binom

from matplotlib import animation, rc

plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'



# initializing plot

def init_plot():

    fig = plt.figure()

    ax = plt.axes(xlim=(0, 1), ylim=(0, 13))

    line, = ax.plot([], [], lw=2)

    ttl = ax.text(0.6,0.8,'', transform = ax.transAxes, va='center', size=20)

    ax.set_xlabel("p")

    ax.set_ylabel("probability density")

    return fig, ax, ttl, line



# random variates

samples = binom.rvs(1,0.4, size=200)



# starting parameters and x values

a = 1

b = 1

x = np.linspace(0.00,1, 100)



# init function

def init():

    ttl.set_text("$\\alpha= " + str(a) + ", \\beta=" + str(b) + "$")

    y =  beta.pdf(x,a,b)

    line.set_data(x,y)

    return line,



# animating the stuff

def animate(i):

    global a,b

    # somehow the init frame is not drawn, so a small hack here

    if i != 0:

        a += samples[i-1]

        b += 1 - samples[i-1]

    ttl.set_text("$\\alpha= " + str(a) + ", \\beta=" + str(b) + "$")

    y =  beta.pdf(x,a,b)

    line.set_ydata(y)

    return line,



# let's animate

fig, ax, ttl, line = init_plot()

anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=200, interval=100, blit=True)

plt.close()

HTML(anim.to_html5_video())
fig, ax, ttl, line = init_plot()

a = 50

b = 10

anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=200, interval=100, blit=True)

plt.close()

HTML(anim.to_html5_video())

fig, ax, ttl, line = init_plot()

a = 4

b = 6

anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=200, interval=100, blit=True)

plt.close()

HTML(anim.to_html5_video())
dist = beta(1+80,1+120)

print("95% Credible interval:", dist.interval(0.95))
binom.cdf(80,200,0.5)
fig, ax = plt.subplots(1,1)

plot_beta(81,121,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
# for tractability work with the logarithm

from scipy.special import betaln



p_alt = betaln(80+1,120+1) - betaln(1,1)

p_null = betaln(80+100,120+100) - betaln(100,100)



bf = p_alt - p_null

print("Log Bayes factor: ", bf)
fig, ax = plt.subplots(1,1)

plot_beta(100.,100.,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(1.,1.,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
fig, ax = plt.subplots(1,1)

plot_beta(1000.,1000.,ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels);
# for tractability work with the logarithm

from scipy.special import betaln



p_alt = betaln(80+1,120+1) - betaln(1,1)

p_null = betaln(80+1000,120+1000) - betaln(1000,1000)



bf = p_alt - p_null

print("Log Bayes factor: ", bf)
print((80/200.)**1)
print((80/200.)**10)
pp = betaln(80+1+1,200-80+1) - betaln(81,121)

print(np.exp(pp))
pp = betaln(80+10+1,200-80+1) - betaln(81,121)

print(np.exp(pp))
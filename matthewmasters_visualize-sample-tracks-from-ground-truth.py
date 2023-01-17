import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trackml.dataset import load_event
import os

path_to_train = "../input/train_1"
event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

def plot_track(axis,particle_id):
    p_traj = (truth[truth.particle_id == particle_id][['tx', 'ty', 'tz']]).sort_values(by='tz')
    axis.plot(
        xs=p_traj.tx,
        ys=p_traj.ty,
        zs=p_traj.tz)

def show_3Dplot(ax):
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z  (mm) -- Detection layers')
    plt.show()

def plot_sample_track(N):
    fig, ax = plt.subplots()
    ax = Axes3D(fig)
    sample_particles = particles.sample(N)
    if N > 1:
        for particle_id in sample_particles.particle_id.values:
            plot_track(ax,particle_id)
    else:
        plot_track(ax,sample_particles.particle_id.values[0])
    show_3Dplot(ax)

plot_sample_track(10)

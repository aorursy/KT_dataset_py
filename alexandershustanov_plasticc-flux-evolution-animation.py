import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft
from statsmodels.tsa.arima_model import ARIMA
import os
print(os.listdir("../input"))
import io
import base64
from IPython.display import HTML
import matplotlib.animation as animation
train_df = pd.read_csv('../input/training_set.csv')
def calc_frames(train_df, object_id):
    object_vals = train_df[train_df.object_id == object_id].copy()
    
    object_vals.mjd = object_vals.mjd.round()
    
    start_time = object_vals.mjd.min()
    end_time = object_vals.mjd.max()
    
    cur_time = start_time - 3
    
    max_frames = int((end_time - start_time)//3)
    frames = np.zeros((max_frames, 6))*np.nan
    
    frame_time = np.zeros(max_frames)

    for i in range(max_frames):
        cur_time = cur_time + 3
        frames[i, :] = np.array([object_vals[(object_vals.passband == b) & ((object_vals.mjd - cur_time).abs() < 3)].pipe(lambda x: np.nan if x.shape[0] == 0 else x.flux.iloc[0])  for b in range(6)])        
        frame_time[i] = cur_time
    
    frame_time = frame_time[~np.isnan(frames).all(axis = 1)]
    frames = frames[~np.isnan(frames).all(axis = 1)]
    
    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = _nan_helper(frames)
    frames[nans]= np.interp(x(nans), x(~nans), frames[~nans])

    return frames, frame_time
fr, tm = calc_frames(train_df, 53925325)
def animate(frames, frame_time, name):
    fig, ax = plt.subplots()

    x = [350, 500, 600, 700, 875, 1000] 
    y = [-1000, 1000, 0,0,0,0]

    line, = ax.plot(x, y, lw=2)
    text = ax.text(0.85, 0.95,  '', transform=ax.transAxes)

    def init():
        line.set_data(x, [np.nan] * 6)
        return line,

    def animate(i):
        line.set_ydata(frames[i, :])
        text.set_text(frame_time[i])
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2,save_count=50, frames=frames.shape[0], blit=True)

    plt.close()

    filename = name
    ani.save(filename, writer='imagemagick', fps=8)
animate(*calc_frames(train_df, 615), "615.gif")
filename = '615.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
animate(*calc_frames(train_df, 53925325), "53925325.gif")
filename = '53925325.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
animate(*calc_frames(train_df, 713), "713.gif")
filename = '713.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
train_df.object_id.unique()
animate(*calc_frames(train_df, 730), "730.gif")
filename = '730.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
animate(*calc_frames(train_df, 130762946), "130762946.gif")
filename = '130762946.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
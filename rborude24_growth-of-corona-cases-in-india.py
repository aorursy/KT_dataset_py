import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import datetime
import matplotlib.colors as mc
import colorsys
from random import randint
df = pd.read_csv('../input/covid.csv')
df.head(10)
current_date = 1
dff = df[df['Date'].eq(current_date)].sort_values(by='Total', ascending=True)
dff
fig, ax = plt.subplots(figsize=(15, 20))
ax.barh(dff['State'], dff['Total'])
fig, ax = plt.subplots(figsize=(36, 20))
dff = dff[::-1]   # flip values from top to bottom
# pass colors values to `color=`
ax.barh(dff['State'], dff['Total'])
# iterate over the values to plot labels and values (Tokyo, Asia, 38194.2)
for i, (value, name) in enumerate(zip(dff['Total'], dff['State'])):
   # ax.text(value, i,     name,            ha='right')  # Tokyo: name
   # ax.text(value, i-.25, group_lk[name],  ha='right')  # Asia: group name
    ax.text(value, i,     value,           ha='left')   # 38194.2: value
# Add year right middle portion of canvas
ax.text(1, 0.4, current_date, transform=ax.transAxes, size=46, ha='right')
fig, ax = plt.subplots(figsize=(36, 20))

def draw_barchart(date):
    dff = df[df['Date'].eq(date)].sort_values(by='Total', ascending=True).tail(10)
    ax.clear()
    normal_colors = dict(zip(df['State'].unique(), rgb_colors_opacity))
    dark_colors = dict(zip(df['State'].unique(), rgb_colors_dark))
    ax.barh(dff['State'], dff['Total'],color = [normal_colors[x] for x in dff['State']], height = 0.8,edgecolor =([dark_colors[x] for x in dff['State']]), linewidth = '6')
    dx = dff['Total'].max() / 200
    for i, (value, name) in enumerate(zip(dff['Total'], dff['State'])):
        ax.text(value-dx, i,     name,           size=26, weight=600, ha='right', va='bottom')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=26, ha='left',  va='center')
    # ... polished styles
    ax.text(0, 1.06, 'Cases (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'The Rise of Corona Virus Cases in India',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'credit @RohitBorude',size=24, transform=ax.transAxes, ha='right',
            color='#c92216')
    plt.box(False)
    
draw_barchart(1)
def transform_color(color, amount = 0.5):

    try:
        c = mc.cnames[color]
    except:
        c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

all_names = df['State'].unique().tolist()
random_hex_colors = []
for i in range(len(all_names)):
    random_hex_colors.append('#' + '%06X' % randint(0, 0xFFFFFF))

rgb_colors = [transform_color(i, 1) for i in random_hex_colors]
rgb_colors_opacity = [rgb_colors[x] + (0.825,) for x in range(len(rgb_colors))]
rgb_colors_dark = [transform_color(i, 1.12) for i in random_hex_colors]
import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(36, 20))
animator = animation.FuncAnimation(fig, draw_barchart, frames=range(0, 45), repeat = False)
#HTML(animator.save('corona.mp4'))
HTML(animator.to_jshtml()) 
# or use animator.to_html5_video() or animator.save() 
#plt.show()

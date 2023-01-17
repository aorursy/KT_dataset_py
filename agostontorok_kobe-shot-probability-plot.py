%pylab inline
import pandas as pd
#from http://savvastjortjoglou.com/nba-shot-sharts.html
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax
df = pd.read_csv('../input/data.csv').loc[:,['loc_x','loc_y','shot_made_flag']]
hist_range = [[df.loc_x.min(), df.loc_x.max()],
              [df.loc_y.min(), df.loc_y.max()]]

bin_num = 50
shots, _, _ = np.histogram2d(df.loc[df.shot_made_flag == 1.0,'loc_x'],
               df.loc[df.shot_made_flag == 1.0,'loc_y'],
              bins = bin_num, range = hist_range)
totals, edges_x, edges_y = np.histogram2d(df.loc[:,'loc_x'],
               df.loc[:,'loc_y'],
              bins = bin_num, range = hist_range)
shot_probability = np.true_divide(shots,totals)
import matplotlib.colors as mcolor
import scipy.stats.mstats as scism

color_map = 'YlOrRd'


shot_probability_w_alpha = plt.get_cmap(color_map)(transpose(fliplr(shot_probability))) # align histograms with the basketball court sketch
totals = scism.winsorize(totals, (.05, .05))
alpha_channel = transpose(fliplr(totals/totals.max()))
shot_probability_w_alpha[:,:,3] = alpha_channel
# this is a workaround here because of matplotlib imshow rgba behaviour https://github.com/matplotlib/matplotlib/issues/3343
# I create the alpha behaviour manually using white background
shot_probability_w_alpha = (shot_probability_w_alpha[..., :-1] * shot_probability_w_alpha[..., -1].reshape(bin_num, bin_num, 1) +
                            (1 - shot_probability_w_alpha[..., -1]).reshape(bin_num, bin_num, 1))

plt.figure(figsize=(12,11))
draw_court(outer_lines=True)
X, Y = np.meshgrid(edges_x, edges_y)
plt.imshow(shot_probability_w_alpha, cmap = color_map, 
           interpolation = 'nearest',
          extent = [edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
plt.xlim(-300,300)
plt.ylim(-100,500)
plt.axis('off')
plt.clim(0,1)
plt.colorbar(shrink = 0.3, label = 'Probability of shot', cmap = color_map)
plt.title('Probability of shot\nOpacity changes with number of tries', size = 20)
plt.show()
# Install:

# Kaggle environments.

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install .



# GFootball environment.

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.7 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.7.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
%%writefile agent.py



# Importing Important Imports

from kaggle_environments.envs.football.helpers import *

import math



class Vector:



	'''

	Vector Object

	Parameters: Iterable of length 2 or 3

	'''



	def __init__(self, positions = [0, 0, 0]):

		if len(positions) < 3: positions.append(0)

		self.x, self.y, self.z = positions

		self.vel = None



	def dist(self, other):

		''' Euclidean distance '''

		return math.hypot(other.x - self.x, other.y - self.y)

	

	def add(self, vel):

		''' Adds one vector to the other '''

		return Vector([self.x + vel.x, self.y + vel.y, self.z + vel.z])



	def mult(self, x):

		''' Scales the vector by x '''

		return Vector([self.x * x, self.y * x, self.z * x])



@human_readable_agent

def agent(obs):



	''' Main Agent '''



	# Loading Variables



	N = len(obs['left_team'])



	# Teams

	team = list(map(Vector, obs['left_team']))

	opponents = list(map(Vector, obs['right_team']))



	# Indexes of Active Players

	baller = obs['ball_owned_player']

	active = obs['active']



	# Key Players

	player = team[active]

	goalkeeper = opponents[0]



	# Ball Variables

	ballOwned = (obs['ball_owned_team'] == 0 and active == baller)

	ball = Vector(obs['ball'])

	ball.vel = Vector(obs['ball_direction'])



	# Special Helpers

	sticky = obs['sticky_actions']

	mode = obs['game_mode']



	# Enemy Goal

	target = Vector([1, 0])



	# Directions for movement

	directions = [

		[Action.TopLeft, Action.Top, Action.TopRight],

		[Action.Left, Action.Idle, Action.Right],

		[Action.BottomLeft, Action.Bottom, Action.BottomRight]

	]



	def stickyCheck(action, direction):

		''' Checking for direction and actions '''

		if direction not in sticky:

			return direction

		return action

	

	def dirsign(value):

		''' Getting index for directions '''

		if abs(value) < 0.01: return 1

		elif value < 0: return 0

		return 2



	def getDirection(target, position = player):

		''' Getting direction to move from position to target '''

		xdir = dirsign(target.x - position.x)

		ydir = dirsign(target.y - position.y)

		return directions[ydir][xdir]



	# Always Sprint

	if Action.Sprint not in sticky:

		return Action.Sprint



	# Offense Patterns

	if ballOwned:



		# Special Situations

		if mode in [GameMode.Penalty, GameMode.Corner, GameMode.FreeKick]:

			if player.x > 0: return Action.Shot

			return Action.LongPass



		# Goalkeeper Check

		if baller == 0: 

			return Action.LongPass

		

		# Bad Angle Pass

		if abs(player.y) > 0.2 and player.x > 0.7: 

			return Action.HighPass

			

		# Close to Goalkeeper Shot

		if player.dist(goalkeeper) < 0.4:

			return Action.Shot



		# Goalkeeper is Out

		if goalkeeper.dist(target) > 0.2:

			if player.x > 0:

				return Action.Shot



		#####################

		## Your Ideas Here ##

		#####################



		# Run to Goal

		return getDirection(target)



	# Defensive Patterns

	else:



		# Find the Ball's Next Positions

		nextBall = ball.add(ball.vel.mult(3))



		# Running to the next Ball Position

		if ball.dist(player) > 0.005:

			return getDirection(nextBall)

		

		# Sliding

		elif ball.dist(player) <= 0.005:

			return Action.Slide

		

		# Running Directly at the Ball

		return getDirection(ball)
%%writefile visualizer.py

from matplotlib import animation, patches, rcParams

from matplotlib import pyplot as plt

from kaggle_environments.envs.football.helpers import GameMode



WIDTH = 110

HEIGHT = 46.2

PADDING = 10





def initFigure(figwidth=12):

    figheight = figwidth * (HEIGHT + 2 * PADDING) / (WIDTH + 2 * PADDING)



    fig = plt.figure(figsize=(figwidth, figheight))

    ax = plt.axes(xlim=(-PADDING, WIDTH + PADDING), ylim=(-PADDING, HEIGHT + PADDING))

    plt.axis("off")

    return fig, ax





def drawPitch(ax):

    paint = "white"



    # Grass around pitch

    rect = patches.Rectangle((-PADDING / 2, -PADDING / 2), WIDTH + PADDING, HEIGHT + PADDING,

                             lw=1, ec="black", fc="#3f995b", capstyle="round")

    ax.add_patch(rect)



    # Pitch boundaries

    rect = plt.Rectangle((0, 0), WIDTH, HEIGHT, ec=paint, fc="None", lw=2)

    ax.add_patch(rect)



    # Middle line

    plt.plot([WIDTH / 2, WIDTH / 2], [0, HEIGHT], color=paint, lw=2)



    # Dots

    dots_x = [11, WIDTH / 2, WIDTH - 11]

    for x in dots_x:

        plt.plot(x, HEIGHT / 2, "o", color=paint, lw=2)



    # Penalty box

    penalty_box_dim = [16.5, 40.3]

    penalty_box_pos_y = (HEIGHT - penalty_box_dim[1]) / 2



    rect = plt.Rectangle((0, penalty_box_pos_y),

                         penalty_box_dim[0], penalty_box_dim[1], ec=paint, fc="None", lw=2)

    ax.add_patch(rect)

    rect = plt.Rectangle((WIDTH, penalty_box_pos_y), -

                         penalty_box_dim[0], penalty_box_dim[1], ec=paint, fc="None", lw=2)

    ax.add_patch(rect)



    # Goal box

    goal_box_dim = [5.5, penalty_box_dim[1] - 11 * 2]

    goal_box_pos_y = (penalty_box_pos_y + 11)



    rect = plt.Rectangle((0, goal_box_pos_y),

                         goal_box_dim[0], goal_box_dim[1], ec=paint, fc="None", lw=2)

    ax.add_patch(rect)

    rect = plt.Rectangle((WIDTH, goal_box_pos_y),

                         -goal_box_dim[0], goal_box_dim[1], ec=paint, fc="None", lw=2)

    ax.add_patch(rect)



    # Goals

    goal_width = 0.044 / 0.42 * HEIGHT

    goal_pos_y = (HEIGHT / 2 - goal_width / 2)

    rect = plt.Rectangle((0, goal_pos_y), -2, goal_width,

                         ec=paint, fc=paint, lw=2, alpha=0.3)

    ax.add_patch(rect)

    rect = plt.Rectangle((WIDTH, goal_pos_y), 2, goal_width,

                         ec=paint, fc=paint, lw=2, alpha=0.3)

    ax.add_patch(rect)



    # Middle circle

    mid_circle = plt.Circle([WIDTH / 2, HEIGHT / 2], 9.15, color=paint, fc="None", lw=2)

    ax.add_artist(mid_circle)



    # Penalty box arcs

    left = patches.Arc([11, HEIGHT / 2], 2 * 9.15, 2 * 9.15,

                       color=paint, fc="None", lw=2, angle=0, theta1=308, theta2=52)

    ax.add_patch(left)

    right = patches.Arc([WIDTH - 11, HEIGHT / 2], 2 * 9.15, 2 * 9.15,

                        color=paint, fc="None", lw=2, angle=180, theta1=308, theta2=52)

    ax.add_patch(right)



    # Arcs on corners

    corners = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]

    angle = 0

    for x, y in corners:

        c = patches.Arc([x, y], 2, 2,

                        color=paint, fc="None", lw=2, angle=angle, theta1=0, theta2=90)

        ax.add_patch(c)

        angle += 90





def scale_x(x):

    return (x + 1) * (WIDTH / 2)





def scale_y(y):

    return (y + 0.42) * (HEIGHT / 0.42 / 2)





def extract_data(raw_obs):

    obs = raw_obs[0]["observation"]["players_raw"][0]

    res = dict()

    res["left_team"] = [(scale_x(x), scale_y(y)) for x, y in obs["left_team"]]

    res["right_team"] = [(scale_x(x), scale_y(y)) for x, y in obs["right_team"]]



    ball_x, ball_y, ball_z = obs["ball"]

    res["ball"] = [scale_x(ball_x), scale_y(ball_y), ball_z]

    res["score"] = obs["score"]

    res["steps_left"] = obs["steps_left"]

    res["ball_owned_team"] = obs["ball_owned_team"]

    res["ball_owned_player"] = obs["ball_owned_player"]

    res["right_team_roles"] = obs["right_team_roles"]

    res["left_team_roles"] = obs["left_team_roles"]

    res["left_team_direction"] = obs["left_team_direction"]

    res["right_team_direction"] = obs["right_team_direction"]

    res["game_mode"] = GameMode(obs["game_mode"]).name

    return res





def draw_team(obs, team, side):

    x_coords, y_coords = zip(*obs[side])

    team.set_data(x_coords, y_coords)





def draw_ball(obs, ball):

    ball.set_markersize(8 + obs["ball"][2])  # Scale size of ball based on height

    ball.set_data(obs["ball"][:2])





def draw_ball_owner(obs, ball_owner):

    if obs["ball_owned_team"] == 0:

        x, y = obs["left_team"][obs["ball_owned_player"]]

        ball_owner.set_data(x, y)

    elif obs["ball_owned_team"] == 1:

        x, y = obs["right_team"][obs["ball_owned_player"]]

        ball_owner.set_data(x, y)

    else:

        ball_owner.set_data([], [])





def draw_team_active(obs, team_left_active, team_right_active):

    team_left_active.set_data(WIDTH / 2 - 20, -7)

    team_right_active.set_data(WIDTH / 2 + 20, -7)



    if obs["ball_owned_team"] == 0:

        team_left_active.set_markerfacecolor("firebrick")

    else:

        team_left_active.set_markerfacecolor("mistyrose")



    if obs["ball_owned_team"] == 1:

        team_right_active.set_markerfacecolor("blue")

    else:

        team_right_active.set_markerfacecolor("lightcyan")





def draw_players_directions(obs, directions, side):

    index = 0

    if "right" in side:

        index = 11

    for i, player_dir in enumerate(obs[f"{side}_direction"]):

        x_dir, y_dir = player_dir

        dist = (x_dir ** 2 + y_dir ** 2)**0.5 + 0.00001  # to prevent division by 0

        x = obs[side][i][0]

        y = obs[side][i][1]

        directions[i + index].set_data([x, x + x_dir / dist], [y, y + y_dir / dist])





steps = None

drawings = None

directions = None

ball = ball_owner = None

team_left = team_right = None

team_left_active = team_right_active = None

text_frame = game_mode = match_info = None





def init():

    ball.set_data([], [])

    ball_owner.set_data([], [])

    team_left.set_data([], [])

    team_right.set_data([], [])

    team_left_active.set_data([], [])

    team_right_active.set_data([], [])

    return drawings





def animate(i):

    obs = extract_data(steps[i])



    # Draw info about ball possesion

    draw_ball_owner(obs, ball_owner)

    draw_team_active(obs, team_left_active, team_right_active)



    # Draw players

    draw_team(obs, team_left, "left_team")

    draw_team(obs, team_right, "right_team")



    draw_players_directions(obs, directions, "left_team")

    draw_players_directions(obs, directions, "right_team")



    draw_ball(obs, ball)



    # Draw textual informations

    text_frame.set_text(f"Step {i}/{obs['steps_left'] + i - 1}")

    game_mode.set_text(f"Game Mode: {obs['game_mode']}")



    score_a, score_b = obs["score"]

    match_info.set_text(f" Left Team {score_a} : {score_b} Right Team")



    return drawings





def visualize(trace):

    global steps

    global drawings

    global directions

    global ball, ball_owner

    global team_left, team_right

    global team_left_active, team_right_active

    global text_frame, game_mode, match_info



    rcParams['font.family'] = 'monospace'

    rcParams['font.size'] = 14



    steps = trace



    fig, ax = initFigure()

    drawPitch(ax)

    ax.invert_yaxis()



    ball_owner, = ax.plot([], [], "o", ms=20, mfc="yellow", alpha=0.5)

    team_left, = ax.plot([], [], "o", ms=12, mfc="firebrick", mew=1, mec="white")

    team_right, = ax.plot([], [], "o", ms=12, mfc="blue", mew=1, mec="white")

    ball, = ax.plot([], [], "o", ms=8, mfc="wheat", mew=1, mec="black")



    team_left_active, = ax.plot([], [], "o", ms=18, mfc="mistyrose", mec="None")

    team_right_active, = ax.plot([], [], "o", ms=18, mfc="lightcyan", mec="None")



    textheight = -6

    text_frame = ax.text(-5, textheight, "")

    match_info = ax.text(WIDTH / 2, textheight, "", ha="center")

    game_mode = ax.text(WIDTH + 5, textheight, "", ha="right")



    # Drawing of directions definitely can be done in a better way

    directions = []

    for _ in range(22):

        direction, = ax.plot([], [], color="yellow", lw=1.5)

        directions.append(direction)



    drawings = [team_left_active, team_right_active, ball_owner, team_left, team_right,

                ball, text_frame, match_info, game_mode]



    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True,

                                   interval=100, frames=len(steps), repeat=False)

    return anim
from kaggle_environments import make

from kaggle_environments import agent

import sys



from visualizer import visualize

from IPython.display import HTML



env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True}, debug = True)

output = env.run(['/kaggle/working/agent.py', 'do_nothing'])

scores = output[-1][0]["observation"]["players_raw"][0]["score"]



print("\nScores  {0} : {1}".format(*scores))



# Cool Visualization

viz = visualize(output)

HTML(viz.to_html5_video())
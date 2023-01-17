#Loading libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls


#Importing the datasets 

players = pd.read_csv("../input/Player.csv", encoding='ISO-8859-1' )
player_match = pd.read_csv("../input/Player_match.csv", encoding='ISO-8859-1' )
team = pd.read_csv("../input/Team.csv", encoding='ISO-8859-1' )
ball_fact = pd.read_csv ("../input/Ball_By_Ball.csv", encoding='ISO-8859-1' )
match = pd.read_csv ("../input/Match.csv", encoding='ISO-8859-1' )
players.head(2)

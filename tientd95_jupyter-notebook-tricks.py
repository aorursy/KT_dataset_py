import pandas as pd
%pylab inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# read data
df = pd.read_csv('../input/google-job-skills/job_skills.csv')
# display multiple outputs
df.head(3)
df.tail(3)
%lsmagic
int_variable = 5
float_variable = 6.1
list_variable = [1,2]
string_variable = 'hi'
string_variable_2 = 'how are you'

%who str        # List all variable with string type
%who list       # List all variable with list type
import numpy as np
from IPython.display import Audio
framerate = 44100
t = np.linspace(0,5,framerate*5)
data = np.sin(2*np.pi*220*t**2)
Audio(data,rate=framerate)
from IPython.display import YouTubeVideo
YouTubeVideo('zGFKSQlef_0')
# Multicursor support with Alt keyboard
from IPython.display import YouTubeVideo
YouTubeVideo('q_FturFMdj0')
from IPython.display import Markdown, display
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

printmd("**bold and blue**", color="blue")
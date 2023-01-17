! pip install bamboolib 
! python -m bamboolib install_extensions 
import bamboolib as bam
import pandas as pd
! wget https://raw.githubusercontent.com/dataprofessor/data/master/nba-player-stats-2019.csv
import pandas as pd
df = pd.read_csv('nba-player-stats-2019.csv')
df
import plotly.express as px
px.histogram(df, x='Pos', y='PTS', histfunc='max')
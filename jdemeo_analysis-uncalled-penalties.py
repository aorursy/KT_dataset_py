import pandas as pd
import numpy as np
# Import data related to uncalled penalties
injury_df = pd.read_csv('../input/ngsconcussion/injury_play.csv')

# Isolate only plays with an uncalled penalty 
injury_df = injury_df[injury_df['uncalled_penalty'] == 1]
injury_df

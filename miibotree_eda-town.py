import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

town_info = pd.read_csv('../input/town_state.csv')
town_info.Agencia_ID.count()
town_num_per_state = town_info.loc[:, ['Town', 'State']].drop_duplicates().State.value_counts()
town_num_per_state
agenda_num_per_state = town_info.loc[:, ['Agencia_ID', 'State']].drop_duplicates().State.value_counts()
agenda_num_per_state
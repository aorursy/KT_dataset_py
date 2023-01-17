import pandas as pd
import plotly.express as px
# YOUR CODE HERE
# YOUR CODE HERE
# YOUR CODE HERE
!pip install bar_chart_race
import bar_chart_race as bcr
# YOUR CODE HERE
# YOUR CODE HERE
bcr.bar_chart_race(DATAFRAME,
                   fixed_max=True,  # This option makes the x axis have a fixed max
                   n_bars=15,       # This option limits the graphic to 15 countries at a time
                   filter_column_colors=True, # This reduces the number of repeated colors
                   period_length=1000 # This makes it so each year takes 1000 ms (1 second)
                  )
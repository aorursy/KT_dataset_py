!pip install pyrugga
# import the library

import pyrugga
# pass file

df = pyrugga.Match('/kaggle/input/8400_irevwal_new.xml')
#print summary of match

df.summary
#list all actions in a matches

df.events
#time line of a match

df.timeline
#prints a summary of each players actions normalise by phases while pitch

df.player_summary(norm='phases')
#prints a heatmap

df.heat_map(event='Carry', event_type='One Out Drive', description='Crossed Gainline')
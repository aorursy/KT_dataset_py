import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

d.index = pd.DatetimeIndex(d.timeStamp)

d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
# Start with Cheltenham

#  





t=d[d.twp == 'CHELTENHAM']

# Not vehicle, fire or traffic

t=t[~t.title.str.match(r'.*VEHICLE.*')] 

t=t[~t.title.str.match(r'.*FIRE.*')]

t=t[~t.title.str.match(r'.*Traffic.*')]



# The school is at the intersection of RICES MILL and PANTHER. It could be reported

# in any order. The school is also on OLD MILL

t=t[(t.desc.str.match(r'.*RICES MILL.*')) & ( t.desc.str.match(r'.*PANTHER.*') ) |

   (t.desc.str.match(r'.*OLD MILL.*'))]

t[['title','desc']]
# Abington

#  HIGHLAND AVE & GHOST RD

#  HIGHLAND AVE & CANTERBURY





t=d[d.twp == 'ABINGTON']

# Not vehicle, fire or traffic

t=t[~t.title.str.match(r'.*VEHICLE.*')] 

t=t[~t.title.str.match(r'.*FIRE.*')]

t=t[~t.title.str.match(r'.*Fire.*')]

t=t[~t.title.str.match(r'.*Traffic.*')]



# The school is at the intersection of HIGHLAND AVE and GHOST. It could be reported

# in any order. The school is also on HIGHLAND AVE and CANTERBURY.

t=t[(t.desc.str.match(r'.*HIGHLAND AVE.*') &  t.desc.str.match(r'.*GHOST.*') ) |

   (t.desc.str.match(r'.*HIGHLAND.*') & t.desc.str.match(r'.*CANTERBURY.*'))]

t[['title','desc']]
# Upper Dublin

# SPARK DR; UPPER DUBLIN

# LOCH ALSH 

t=d[d.twp == 'UPPER DUBLIN']

# Not vehicle, fire or traffic

t=t[~t.title.str.match(r'.*VEHICLE.*')] 

t=t[~t.title.str.match(r'.*FIRE.*')]

t=t[~t.title.str.match(r'.*Fire.*')]

t=t[~t.title.str.match(r'.*Traffic.*')]



# The school is at the intersection of LOCH ALSH and SPARK. It could be reported



t=t[(t.desc.str.match(r'.*LOCH ALSH .*') & t.desc.str.match(r'.*SPARK DR.*')  ) |

   (t.desc.str.match(r'.*SPARK DR.*') )]

t[['title','desc']]
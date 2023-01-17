import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as date_parser
comments = pd.read_csv("../input/result.csv")
comments = comments.drop(['paging__cursors__before','paging__cursors__after' ], axis=1)
comments.describe()
comments.head(5)
times = comments['data__created_time'].apply(lambda x: date_parser.parse(x)).sort_values()
plt.plot(range(1,times.size + 1), times)
plt.show()
plt.plot(range(0,30), times.head(30))
plt.show()
times.head(30)


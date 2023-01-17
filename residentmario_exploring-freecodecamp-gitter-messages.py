import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
chat = pd.read_csv("../input/freecodecamp_casual_chatroom.csv", index_col=0)
fig_kwargs = {'figsize': (12, 6), 'fontsize': 16}
chat = chat.iloc[:, 1:]
chat.head(3)
chat.sent.duplicated().sum() / len(chat)
chat = chat.drop_duplicates()
chat = chat.loc[~chat.sent.duplicated()]
chat['fromUser.displayName'].value_counts().head(10).plot.bar(**fig_kwargs, 
                                                              title="Top 10 Posters")
(pd.to_datetime(chat[chat['fromUser.displayName'] == "Serenity"].sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(**fig_kwargs, title="Monthly Posts by A Top Poster"))
chat[chat['fromUser.staff'].notnull()].head()
chat = chat.assign(sent=pd.to_datetime(chat.sent))
ax = chat.sent.to_frame().assign(n=0).set_index('sent').resample('M').count().plot.line(
    **fig_kwargs, title="freeCodeCamp Gitter Chat Activity over Time"
)

(pd.to_datetime(chat[chat['fromUser.displayName'] == "Serenity"].sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(ax=ax))
import numpy as np
chat.groupby(pd.Grouper(key='sent', 
                        freq='M')).readBy.agg([np.median, 
                                               np.min, 
                                               np.max]).plot.line(**fig_kwargs, 
                                                                  title="freeCodeCamp Gitter Views Per Message")
ax = plt.gca()
ax.set_ylim([0, 100])
chat.loc[chat.readBy.sort_values(ascending=False).head(10).index].text
chat.readBy.sort_values(ascending=False).head(10).mean()
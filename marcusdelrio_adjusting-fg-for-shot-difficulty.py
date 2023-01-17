# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
rd=pd.read_csv("../input/shot_logs.csv",low_memory=False)

rd.loc[:,"game_date"]=pd.DatetimeIndex(rd.MATCHUP.str[:12])
#chart the fall in FG pct by shot distance

shot_dist_pivot=rd.pivot_table(values=["FGM"],index="SHOT_DIST",aggfunc=[len,np.sum])

shot_dist_plot_data=shot_dist_pivot[("sum","FGM")].div(shot_dist_pivot[("len","FGM")])

shot_dist_plot_data[:25].plot(title="FG % by Shot Dist")
#chart the increase in FG pct by shot clock time

shot_clock_pivot=rd.pivot_table(values=["FGM"],index="SHOT_CLOCK",aggfunc=[len,np.sum])

shot_clock_plot_data=shot_clock_pivot[("sum","FGM")].div(shot_clock_pivot[("len","FGM")])

shot_clock_plot_data.plot(title="FG % by Shot Clock Time Remaining")
#chart the FG pct by defender distance

closest_def_pivot=rd.pivot_table(values=["FGM"],index="CLOSE_DEF_DIST",aggfunc=[len,np.sum])

closest_def_pivot.columns=closest_def_pivot.columns.get_level_values(0)

closest_def_pivot=closest_def_pivot.rename(columns={"len":"shot attempts","sum":"shots made"})

closest_def_pivot["distance"]=np.ceil(closest_def_pivot.index)

closest_def_grouped=closest_def_pivot.groupby("distance").sum()

closest_def_grouped["FgPct"]=closest_def_grouped["shots made"].div(closest_def_grouped["shot attempts"])

closest_def_grouped.FgPct[:15].plot(title="FG % by Closest Defender Distance")
#regression creation

logit_vars=rd[["SHOT_DIST","SHOT_CLOCK","PERIOD","CLOSE_DEF_DIST","FGM"]]

logit_vars=logit_vars.dropna()

logit_vars_X=logit_vars[["SHOT_DIST","SHOT_CLOCK","PERIOD","CLOSE_DEF_DIST"]]

logit_vars_Y=logit_vars["FGM"]

logistic=linear_model.LogisticRegression()

logistic.fit(logit_vars_X,logit_vars_Y)

fitted=logistic.fit(logit_vars_X,logit_vars_Y)

print (fitted.coef_)
#append the predicted shot difficulty on to our full shot dataset

rd2=rd.dropna(subset=["SHOT_DIST","SHOT_CLOCK","PERIOD","CLOSE_DEF_DIST"])

estimated_shot_difficulty=fitted.predict_proba(rd2[["SHOT_DIST","SHOT_CLOCK","PERIOD","CLOSE_DEF_DIST"]])

rd2=rd2.reindex(range(len(rd2)))

to_concat=[x[1] for x in estimated_shot_difficulty]

to_concat=pd.Series(to_concat,index=range(len(to_concat)))

rd2.loc[:,"adjusted_difficulty"]=to_concat
rd2.loc[:,"added_value"]=(rd2.FGM-rd2.adjusted_difficulty)*rd2.PTS_TYPE

pivoted=rd2.pivot_table(values=["added_value"],index=["player_name"])

added_value_top_10=pivoted.sort_values(by="added_value",ascending=False).head(10)

print (added_value_top_10)
rd2_player_slice=rd2[rd2.player_name=="lebron james"]

weekly_added_value=rd2_player_slice.pivot_table(values="added_value",index="game_date")

weekly_added_value.plot()
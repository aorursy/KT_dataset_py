import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here

# 计算数据集winner列中各元素出现的次数
winner = chess_games.winner.value_counts()
print('winner =\n',winner)

# 对比赛场数进行求和并使用pandas.Series.map()函数实现胜率计算
print('winner.sum() =',winner.sum())
winner_radio = winner.map(lambda w: w / winner.sum())
print('winner_radio =\n',winner_radio)
check_q1(winner_radio)
# Your code here
open_name=chess_games.opening_name.map(lambda s:s.split(":")[0].split("|")[0].split("#")[0].strip())
print(open_name)
print(open_name.value_counts())
check_q2(open_name.value_counts())
# Your code here
check_q3(chess_games.assign(n=0).groupby(['white_id','victory_status']).n.apply(len).reset_index())
chess_games.assign(n=0).groupby(['white_id','victory_status']).n.apply(len).reset_index()
chess_games.assign(n=0).groupby(['white_id','victory_status']).size()
chess_games.assign(n=0).groupby(['white_id','victory_status']).n.size()
# 根据阅读参考代码所得，重写实现代码
# 取出组合'white_id','victory_status'后的数据
games_grouped = chess_games.groupby(['white_id','victory_status'])
games_grouped.size()# 返回一个含有分组大小的Series，最后一列才是最终的数据
# 需要注意的是，在group的过程中，已经进行了同一white_id的分组，并且已经将相同的victory_status结合在了一起
#方法一
games_whiteplayer_1 = games_grouped.size().reset_index().rename(columns={0:'n'}) #已经准换为DataFrame数据对象后对列重新命名
#查看所得数据
games_whiteplayer_1 #第一种方法得到的最终数据
# 对重新编程的结果进行检验
check_q3(games_whiteplayer_1)
#方法二
games_whiteplayer2 = games_grouped.size()
games_whiteplayer2.name = 'n' # 先指定Series数据对象的名称为'n'
# 设置Series的index从而将其转化为DataFrame，已经设置过Series的name为'n',故此时该数据列的列名无需再重命名
games_whiteplayer_2 =  games_whiteplayer2.reset_index()
# 查看所得数据
games_whiteplayer_2 # 第二种方法得到的最终数据
# 对重新编程的结果进行检验
check_q3(games_whiteplayer_2)
# 找出在所有比赛中出现次数最多的棋手名字以及出现次数
chess_games.white_id.value_counts().head(20)
# 提取出棋手的名字，在Series数据对象中为Series的index
player_first20 = chess_games.white_id.value_counts().head(20).index
player_first20
# 在Exercise3所生成的数据中寻找具有这些index的数据
# 此处代码执行效率偏低，但便于理解
loc_index = []
for i in range(games_whiteplayer_1.shape[0]):
    if(games_whiteplayer_1.white_id.isin(player_first20)[i]):
        loc_index.append(i)
#games_whiteplayer_1.pipe(lambda df:df.loc[])
loc_index
games_whiteplayer_1.loc[loc_index]
check_q4(games_whiteplayer_1.loc[loc_index])
# 通过链式方法可以进行更快地操作,pipe()函数如同map()和apply()函数类似
games_whiteplayer_1.pipe(lambda df: df.loc[df.white_id.isin(player_first20)])
# Your code here
# 参考代码
answer_q4()
ans4 = chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
ans4
check_q4(ans4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# 对数据先进行拆分重组并自动进行计数，最终发现rowid列的数据就是目标数据，故取出该列并重命名为`n`
times_status = kepler.groupby(['koi_pdisposition','koi_disposition']).count().rowid.rename('n')
check_q5(times_status)
# Your code here
answer_q5()
ans5 = kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
ans5
check_q5(ans5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
wine_reviews.head()
ramen_reviews
# Your code here
# 参考代码：虽然题中说将数据尺寸缩放到1-5星，但实际上是缩放到了0-5星
answer_q6()
ans6 = ((wine_reviews['points'].dropna() - 80) / 4).value_counts().sort_index().rename_axis("Wine Ratings")
check_q6(ans6)
ans6
# Your code here
answer_q7()
ramen_ratings = ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64').value_counts().rename_axis('Ramen Reviews').sort_index()
check_q7(ramen_ratings)
# Your code here
answer_q8()
ans8 = ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').map(lambda v: round(v * 2) / 2).value_counts().rename_axis("Ramen Reviews").sort_index()
check_q8(ans8)
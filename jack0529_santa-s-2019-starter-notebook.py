# このPython 3環境には、多くの役立つ分析ライブラリがインストールされています

# 詳細は右にて記載されています kaggle/python docker image: https://github.com/kaggle/docker-python

# 例えば、下記のようなパッケージが使用できます 



import numpy as np # 線形代数

import pandas as pd # データ整形, CSV file I/O (e.g. pd.read_csv)



# ファイルの入力はこのようにできます "../input/" directory.

# このセルを実行してください (Shift+Enterで実行できます) 指定したディレクトリのすべてのファイルがリストアップされます



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# あなたはすべてのファイルをこのディレクトリに書き込む必要があります
fpath = '/kaggle/input/santa-2019-workshop-scheduling/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-2019-workshop-scheduling/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
# family_id:家族ID、choice_0～9:1から100日目のどの日に開催されるワークショップに参加したいか、n_people：家族の人数

data.head()
submission.head()
family_size_dict = data[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# from 100 to 1

days = list(range(N_DAYS,0,-1))
def cost_function(prediction):



    penalty = 0



    # このディクショナリを使用して、各日の人数を数えます

    daily_occupancy = {k:0 for k in days}

    

    # 全家族をループ処理; d は各家族 f のワ－クショップ日です

    for f, d in enumerate(prediction):



        # 上で作成したディクショナリを使用して、変数に値を代入する

        n = family_size_dict[f]

        choice_0 = choice_dict['choice_0'][f]

        choice_1 = choice_dict['choice_1'][f]

        choice_2 = choice_dict['choice_2'][f]

        choice_3 = choice_dict['choice_3'][f]

        choice_4 = choice_dict['choice_4'][f]

        choice_5 = choice_dict['choice_5'][f]

        choice_6 = choice_dict['choice_6'][f]

        choice_7 = choice_dict['choice_7'][f]

        choice_8 = choice_dict['choice_8'][f]

        choice_9 = choice_dict['choice_9'][f]



        # 家族人数をそれぞれの日の参加数に加算する

        daily_occupancy[d] += n



        # ペナルティを計算する

        if d == choice_0:

            penalty += 0

        elif d == choice_1:

            penalty += 50

        elif d == choice_2:

            penalty += 50 + 9 * n

        elif d == choice_3:

            penalty += 100 + 9 * n

        elif d == choice_4:

            penalty += 200 + 9 * n

        elif d == choice_5:

            penalty += 200 + 18 * n

        elif d == choice_6:

            penalty += 300 + 18 * n

        elif d == choice_7:

            penalty += 300 + 36 * n

        elif d == choice_8:

            penalty += 400 + 36 * n

        elif d == choice_9:

            penalty += 500 + 36 * n + 199 * n

        else:

            penalty += 500 + 36 * n + 398 * n



    # すべての日の参加者をチェックする

    #  (ハードな制約ではなくソフトな制約を使用する)

    for _, v in daily_occupancy.items():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            penalty += 100000000



    # 会計コストを計算する

    # はじめの日の値を計算する(day 100)

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    # 　１日の参加者が125より少ない場合に会計コストがマイナスとなってしまうため、MAXをとる

    accounting_cost = max(0, accounting_cost)

    

    # 残りの日の会計コストを計算する

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return penalty
# サンプルファイルの情報を試しに使ってみる

best = submission['assigned_day'].tolist()

start_score = cost_function(best)



new = best.copy()

# 各家族をループ処理する

for fam_id, _ in enumerate(best):

    # 家族毎の選択を更新

    for pick in range(10):

        day = choice_dict[f'choice_{pick}'][fam_id]

        temp = new.copy()

        temp[fam_id] = day # 新しい参加日を代入

        if cost_function(temp) < start_score:

            new = temp.copy()

            start_score = cost_function(new)



submission['assigned_day'] = new

score = cost_function(new)

submission.to_csv(f'submission_{score}.csv')

print(f'Score: {score}')
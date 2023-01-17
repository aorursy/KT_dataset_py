

#Import　Library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Padas Setting

pd.options.display.max_columns = 50

pd.options.display.max_rows = 400



#Question

"""

※Plusは自由

1.失業と教育水準の関係を可視化

2.『精神病と認識しているか否か』が雇用に与える影響

3.症状/副作用が雇用に与える影響

Plus.社会福祉プログラムの有効性

"""



#Loading CSV

data = pd.read_csv("mental_illness_survey.csv")



#Data Columns

#print(data.columns)



#Heading Data

#print(data.head())

#->1行目は不必要



"""

1.失業と教育水準の関係

'Education'-'I am unemployed'

'Education'-'I am currently employed at least part-time'

"""

"""

2.精神病と認識しているか否かで現在の状況は変化するのか

'I identify as having a mental illness'-'I am unemployed'

'I identify as having a mental illness'-'I am currently employed at least part-time'

"""

"""

3.精神疾患が雇用に与える影響

Lack of concentration-集中力の欠如

Anxiety-不安

Depression-うつ病

Obsessive thinking-強迫観念

Mood swings-気分のむら

Panic attacks-パニック発作

Compulsive behavior-強迫行動

Tiredness-疲れ

'my illness'-'I am unemployed'

'my illness'-'I am currently employed at least part-time'

"""

"""

Plus.社会福祉プログラムの有効性

フードスタンプ・・・無料の食料クーポン

※フードスタンプを受ける人は一食に1.25ドルの予算

※スタンプの不正受給問題

参考：https://macaro-ni.jp/6064

セクション8・・・サポート対象者が暮らす地域の管轄機関が対象者の家主に対して賃貸の一部を支払ってくれる

※低所得・身体障碍・年配が対象

※低所得における基本的な基準は、収入は自分が暮らす地域の『平均所得よりも50%以下』であること

参考1：https://affordablehousingonline.com/section-8-housing

参考2：http://www.kerealtyconsulting.com/2018/04/17/%E3%82%A2%E3%83%A1%E3%83%AA%E3%82%AB%E3%81%AE%E5%AE%B6%E8%B3%83%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E5%88%B6%E5%BA%A6%E3%80%81%E3%82%BB%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B38%E3%81%A8%E3%81%AF%EF%BC%9F/

'Annual income from social welfare programs'-'I am unemployed'

'I receive food stamps'-'Household Income'

'I am on section 8 housing'-'Household Income'

"""



#1.失業と教育水準の関係

education = data["Education"]

unemployer = data["I am unemployed"]

parttimer = data["I am currently employed at least part-time"]

"""

'High School or GED'-高校卒業またはGED所持者(高校卒業資格)

'Some Phd'-博士課程

'Completed Undergraduate'-学部課程修了

'Some Undergraduate'-学部課程

'Some Masters'-修士課程

'Completed Masters'-修士課程修了

'Completed Phd'-博士課程修了

'Some highschool'-高校在学

"""

"""

上記区分の失業者数と現在の状況を調べてみる

--------------------------------------------------------------------------------

'High School or GED'の失業者数 or 現在の状況

.

.

.

'Some highschool'の失業者数 or 現在の状況

--------------------------------------------------------------------------------

[手順]

1.Some\xa0Mastersの'\xa0'を半角スペースに変換

2.一行目を削除

3.それぞれ整形して、一つに繋げる（education_unemployer）

4.education_unemployerを引数に取る関数を定義

5.それぞれの区分でYes/Noを振り分ける

"""

#1.Some\xa0Mastersの'\xa0'を半角スペースに変換

education = data["Education"].replace("Some\xa0Masters","Some Master")

#2.一行目を削除

education = education.drop(0, axis=0)

unemployer = unemployer.drop(0, axis=0)

parttimer = parttimer.drop(0, axis=0)

#3.それぞれ整形して、一つに繋げる（education_unemployer）

education_unemployer = pd.concat([education, unemployer, parttimer], axis=1)

#4.education_unemployerを引数に取る関数を定義

#5.それぞれの区分でYes/Noを振り分ける

def plot_education_unemployment(df, select="unemployer"):

    education_unique = df["Education"].unique()

    per_list = []

    yes_list = []

    no_list = []

    for value in education_unique:

        if select == "unemployer":

            yes_people = df[(df["I am unemployed"]=="Yes") & (df["Education"] == value)][["Education", "I am unemployed"]]

            no_people = df[(df["I am unemployed"]=="No") & (df["Education"] == value)][["Education", "I am unemployed"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            unemployer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(unemployer_per)

        elif select == "parttimer":

            yes_people = df[(df["I am currently employed at least part-time"]=="Yes") & (df["Education"] == value)][["Education", "I am currently employed at least part-time"]]

            no_people = df[(df["I am currently employed at least part-time"]=="No") & (df["Education"] == value)][["Education", "I am currently employed at least part-time"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            parttimer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(parttimer_per)

    labels = [education+"(Yes is {}%)".format(per) for education, per in zip(education_unique, per_list)]

    p1 = plt.barh(range(1, 9), yes_list, align="center", color="indianred")

    p2 = plt.barh(range(1, 9), no_list, align="center", left=yes_list, tick_label=labels, color="skyblue")

    plt.subplots_adjust(left=0.25)

    plt.legend((p1[0], p2[0]), ("{}=Yes".format(select), "{}=No".format(select)), loc="upper right")

    plt.title("What is your educational background?", fontsize=21)

    plt.show()

#plot_education_unemployment(education_unemployer, select="unemployer")

#plot_education_unemployment(education_unemployer, select="parttimer")



#2.精神病と認識しているか否かで現在の状況は変化するのか

identify = data["I identify as having a mental illness"]

unemployer = data["I am unemployed"]

parttimer = data["I am currently employed at least part-time"]

"""

Yes - 自分が精神病だと認識している

No - 自分が精神病だと認識していない

"""

"""

上記区分の失業者数と現在の状況を調べてみる

--------------------------------------------------------------------------------

'精神病と自覚している人'の失業者数 or 現在の状況

'精神病と自覚していない人'の失業者数 or 現在の状況

--------------------------------------------------------------------------------

[手順]

1.一行目を削除

2.Yes -> identify | No -> not identify に変換

3.それぞれ整形して、一つに繋げる（identify_unemployer）

4.identify_unemployerを引数に取る関数を定義

5.それぞれの区分でYes/Noを振り分ける

"""

#1.一行目を削除

identify = identify.drop(0, axis=0)

unemployer = unemployer.drop(0, axis=0)

parttimer = parttimer.drop(0, axis=0)

#2.Yes -> identify | No -> not identify に変換

identify = identify.replace("Yes", "identify").replace("No", "not identify")

#3.それぞれ整形して、一つに繋げる（identify_unemployer）

identify_unemployer = pd.concat([identify, unemployer, parttimer], axis=1)

#4.identify_unemployerを引数に取る関数を定義

#5.それぞれの区分でYes/Noを振り分ける

def plot_identify_unemployment(df, select="unemployer"):

    identify_unique = df["I identify as having a mental illness"].unique()

    per_list = []

    yes_list = []

    no_list = []

    for value in identify_unique:

        if select == "unemployer":

            yes_people = df[(df["I am unemployed"]=="Yes") & (df["I identify as having a mental illness"] == value)][["I identify as having a mental illness", "I am unemployed"]]

            no_people = df[(df["I am unemployed"]=="No") & (df["I identify as having a mental illness"] == value)][["I identify as having a mental illness", "I am unemployed"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            unemployer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(unemployer_per)

        elif select == "parttimer":

            yes_people = df[(df["I am currently employed at least part-time"]=="Yes") & (df["I identify as having a mental illness"] == value)][["I identify as having a mental illness", "I am currently employed at least part-time"]]

            no_people = df[(df["I am currently employed at least part-time"]=="No") & (df["I identify as having a mental illness"] == value)][["I identify as having a mental illness", "I am currently employed at least part-time"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            parttimer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(parttimer_per)

    labels = [identify+"(Yes is {}%)".format(per) for identify, per in zip(identify_unique, per_list)]

    p1 = plt.barh([1, 2], yes_list, align="center", color="indianred")

    p2 = plt.barh([1, 2], no_list, align="center", left=yes_list, tick_label=labels, color="skyblue")

    plt.subplots_adjust(left=0.2)

    plt.legend((p1[0], p2[0]), ("{}=Yes".format(select), "{}=No".format(select)), loc="upper right")

    plt.title("Are you identify as having a mental illness?", fontsize=21)

    plt.show()

#plot_identify_unemployment(identify_unemployer, select="unemployer")

#plot_identify_unemployment(identify_unemployer, select="parttimer")



#3.精神疾患が雇用に与える影響

myillness = data[["I have one of the following issues in addition to my illness", "Unnamed: 28", "Unnamed: 29", "Unnamed: 30", "Unnamed: 31", "Unnamed: 32", "Unnamed: 33", "Unnamed: 34"]]

unemployer = data["I am unemployed"]

parttimer = data["I am currently employed at least part-time"]

"""

I have one of the following issues in addition to my illness -> Lack of concentration-集中力の欠如

Unnamed: 28 -> Anxiety-不安

Unnamed: 29 -> Depression-うつ病

Unnamed: 30 -> Obsessive thinking-強迫観念

Unnamed: 31 -> Mood swings-気分のむら

Unnamed: 32 -> Panic attacks-パニック発作

Unnamed: 33 -> Compulsive behavior-強迫行動

Unnamed: 34 -> Tiredness-疲れ

'my illness'-'I am unemployed'

'my illness'-'I am currently employed at least part-time'

"""

"""

上記区分の失業者数と現在の状況を調べてみる

--------------------------------------------------------------------------------

'Lack of concentration'の失業者数 or 現在の状況

.

.

.

'Tiredness'の失業者数 or 現在の状況

--------------------------------------------------------------------------------

[手順]

1.上記に倣ってコラム名を変換

2.一行目を削除

3.欠損値を0で埋める

4.各列をダミー変数化

5.それぞれ整形して、一つに繋げる（myillness_unemployer）

6.myillness_unemployerを引数に取る関数を定義

7.それぞれの区分でYes/Noを振り分ける

"""

#1.上記に倣ってコラム名を変換

myillness.columns = ["Lack of concentration","Anxiety","Depression","Obsessive thinking","Mood swings","Panic attacks","Compulsive behavior","Tiredness"]

#2.一行目を削除

myillness = myillness.drop(0, axis=0)

unemployer = unemployer.drop(0, axis=0)

parttimer = parttimer.drop(0, axis=0)

#3.欠損値を0で埋める

myillness = myillness.fillna(0)

#4.各列をダミー変数化

dummy_func = lambda x: 1 if x else 0

myillness = myillness.applymap(dummy_func)

#5.それぞれ整形して、一つに繋げる（myillness_unemployer）

myillness_unemployer = pd.concat([myillness, unemployer, parttimer], axis=1)

#6.myillness_unemployerを引数に取る関数を定義

def plot_myillness_unemployment(df, select="unemployer"):

    myillness_unique = [col for col in df.columns.values if not col in "I am unemployed I am currently employed at least part-time"]

    per_list = []

    yes_list = []

    no_list = []

    for value in myillness_unique:

        if select == "unemployer":

            yes_people = df[(df["I am unemployed"]=="Yes") & (df[value] == 1)][[value, "I am unemployed"]]

            no_people = df[(df["I am unemployed"]=="No") & (df[value] == 1)][[value, "I am unemployed"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            unemployer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(unemployer_per)

        elif select == "parttimer":

            yes_people = df[(df["I am currently employed at least part-time"]=="Yes") & (df[value] == 1)][[value, "I am currently employed at least part-time"]]

            no_people = df[(df["I am currently employed at least part-time"]=="No") & (df[value] == 1)][[value, "I am currently employed at least part-time"]]

            yes_num = len(yes_people)

            no_num = len(no_people)

            yes_list.append(yes_num)

            no_list.append(no_num)

            parttimer_per = round(((yes_num / (yes_num+no_num)) * 100), 1)

            per_list.append(parttimer_per)

    labels = [myillness+"(Yes is {}%)".format(per) for myillness, per in zip(myillness_unique, per_list)]

    p1 = plt.barh(range(1, 9), yes_list, align="center", color="indianred")

    p2 = plt.barh(range(1, 9), no_list, align="center", left=yes_list, tick_label=labels, color="skyblue")

    plt.subplots_adjust(left=0.2)

    plt.legend((p1[0], p2[0]), ("{}=Yes".format(select), "{}=No".format(select)), loc="upper right")

    plt.title("What is your mental illness?", fontsize=21)

    plt.show()

#plot_myillness_unemployment(myillness_unemployer, select="unemployer")

#plot_myillness_unemployment(myillness_unemployer, select="parttimer")
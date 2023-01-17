!pip install lifelines
# Load libraries

import re

import glob

import pandas as pd

import pandasql

import numpy as np

import matplotlib.pyplot as plt

import lifelines

from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm

from scipy.stats import ks_2samp



# Scrape for wherever Kaggle decided to put my input files

def find_file(start=""):

    return glob.glob("../input/**/{0}*.csv".format(start), recursive=True)[0]



# Useful shorthand for running SQL

def sql(query):

    return pandasql.sqldf(query, globals())



# Load data from the disk

data = pd.read_csv(find_file("Head_50000_"), sep='\t')



# Cleaning out null-only data

data[data == "None"] = np.nan

data.dropna(axis="columns", how="all")



# Renaming time to be SQL-friendly

data = data.rename(columns={"client_time.1": "time"})

data["time"] = pd.to_datetime(data["time"])



# Converting a few columns from strings to numbers

data["money"] = data["money"].map(float)

data["achievement"] = data["achievement"].map(float)



# Some derived columns based on time

data["seconds"] = (data["time"] - data.groupby("session_id")["time"].transform(lambda x: x.min())).map(lambda x: x.seconds)

data["minutes"] = data["seconds"].map(lambda x: int(x/60))



# Counting the length of buy hovers, easier in Python than SQL

data["len_buy_hovers"] = data["buy_hovers"].str.count("], ") + data["buy_hovers"].str.count("]]")



# Writing out the speaker and line in plain English

data[["speaker", "line"]] = sql("""

    select

        case

            when event_custom in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) then "Human"

            when event_custom = 16 and manual = 1 then "Human"

            else "PC"

        end as speaker,

        case

            when event_custom = 0 then "State(" || money || ", " || speed || ", " || raining || ")"

            when event_custom = 1 and continue = 0 then "Start(New Game)"

            when event_custom = 1 and continue = 1 then "Start(Continue)"

            when event_custom = 2 then "Tutorial(" || event_label || ", " || event_category || ")"

            when event_custom = 3 then "SelectTile(" || tile || ")"

            when event_custom = 4 then "SelectFarmBit(" || farmbit || ")"

            when event_custom = 5 then "SelectItem(" || item || ")"

            when event_custom = 6 and buy = 0 then "SelectBuy(Null)"

            when event_custom = 6 and buy = 1 then "SelectBuy(Home)"

            when event_custom = 6 and buy = 2 then "SelectBuy(Food)"

            when event_custom = 6 and buy = 3 then "SelectBuy(Farm)"

            when event_custom = 6 and buy = 4 then "SelectBuy(Fertilizer)"

            when event_custom = 6 and buy = 5 then "SelectBuy(Livestock)"

            when event_custom = 6 and buy = 6 then "SelectBuy(Skim)"

            when event_custom = 6 and buy = 7 then "SelectBuy(Sign)"

            when event_custom = 6 and buy = 8 then "SelectBuy(Road)"

            when event_custom = 7 and success in (1, "True") and buy = 0 then "Buy(Null, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 1 then "Buy(Home, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 2 then "Buy(Food, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 3 then "Buy(Farm, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 4 then "Buy(Fertilizer, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 5 then "Buy(Livestock, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 6 then "Buy(Skim, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 7 then "Buy(Sign, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (1, "True") and buy = 8 then "Buy(Road, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 0 then "BuyFail(Null, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 1 then "BuyFail(Home, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 2 then "BuyFail(Food, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 3 then "BuyFail(Farm, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 4 then "BuyFail(Fertilizer, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 5 then "BuyFail(Livestock, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 6 then "BuyFail(Skim, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 7 then "BuyFail(Sign, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 7 and success in (0, "False") and buy = 8 then "BuyFail(Road, "  || tile || ", " || len_buy_hovers || ")"

            when event_custom = 8 then "BuyCancel(" || selected_buy || ", " || curr_money || ", " || len_buy_hovers || ")"

            when event_custom = 9 then "RoadBuilds(" || road_builds || ")"

            when event_custom = 10 then "TileUseSelect(" || tile || ", " || marks || ")"

            when event_custom = 11 then "ItemUseSelect(" || item || ")"

            when event_custom = 12 and to_state in (1, "True") then "OpenMenu(Nutrient)"

            when event_custom = 12 and to_state in (0, "False") then "CloseMenu(Nutrient)"

            when event_custom = 13 and shop_open in (1, "True") then "OpenMenu(Shop)"

            when event_custom = 13 and shop_open in (0, "False") then "CloseMenu(Shop)"

            when event_custom = 14 and achievements_open in (1, "True") then "OpenMenu(Achievements)"

            when event_custom = 14 and achievements_open in (0, "False") then "CloseMenu(Achievements)"

            when event_custom = 15 then "SkipTutorial"

            when event_custom = 16 and cur_speed = 0 then "Speed(null)"

            when event_custom = 16 and cur_speed = 1 then "Speed(Pause)"

            when event_custom = 16 and cur_speed = 2 then "Speed(Play x1)"

            when event_custom = 16 and cur_speed = 3 then "Speed(Play x4)"

            when event_custom = 16 and cur_speed = 4 then "Speed(Play x16)"

            when event_custom = 17 and achievement in (0, 1, 2, 3) then "Achieve(Population #" || (achievement % 4 + 1) || ")"

            when event_custom = 17 and achievement in (4, 5, 6, 7) then "Achieve(Farm #" || (achievement % 4 + 1) || ")"

            when event_custom = 17 and achievement in (8, 9, 10, 11) then "Achieve(Money #" || (achievement % 4 + 1) || ")"

            when event_custom = 17 and achievement in (12, 13, 14, 15) then "Achieve(Bloom #" || (achievement % 4 + 1) || ")"

            when event_custom = 18 then "FarmBitDeath"

            when event_custom = 23 then "EndGame"

            when event_custom = 24 and emote_enum = 0 then "Emote(Null)"

            when event_custom = 24 and emote_enum = 1 then "Emote(Hungry)"

            when event_custom = 24 and emote_enum = 2 then "Emote(Very Hungry)"

            when event_custom = 24 and emote_enum = 3 then "Emote(Very Tired)"

            when event_custom = 24 and emote_enum = 4 then "Emote(Bored)"

            when event_custom = 24 and emote_enum = 5 then "Emote(Ennui)"

            when event_custom = 24 and emote_enum = 6 then "Emote(Puke)"

            when event_custom = 24 and emote_enum = 7 then "Emote(Yum)"

            when event_custom = 24 and emote_enum = 8 then "Emote(Tired)"

            when event_custom = 24 and emote_enum = 9 then "Emote(Happy)"

            when event_custom = 24 and emote_enum = 10 then "Emote(Swim)"

            when event_custom = 24 and emote_enum = 11 then "Emote(Sale)"            

            when event_custom = 25 then "FarmFail"

            when event_custom = 26 then "Bloom"

            when event_custom = 27 then "FarmHarvested(" || marks || ")"

            when event_custom = 28 then "MilkProduced(" || marks || ")"

            when event_custom = 29 then "PoopProduced(" || marks || ")"

            when event_custom = 31 then "NewFarmBit"

            else "Other(" || event_custom || ")"

        end as line

    from data

""")



# A version of the line with the time stamp included

data["timed_line"] = sql("""

    select time || "." || line as timed_line

    from data

""")



# Lagged versions of both kinds of lines

data[["lag_line", "lag_timed_line"]] = sql("""

    select

        group_concat(line, "; ") over (partition by session_id order by time rows 4 preceding) as lag_line,

        group_concat(timed_line, "; ") over (partition by session_id order by time rows 4 preceding) as lag_timed_line

    from data

""")



# Save the cleaned up data to the disk, and a random sample too, so we can validate in n-coder

data.to_csv("clean.csv", header=True)

sql("select * from data order by random() limit 2500").to_csv("clean-sample.csv", header=True)
# TODO

def regexp(*exprs):

    regs = [re.compile(expr) for expr in exprs]

    def temp(item):

        if item is not None:

            for reg in regs:

                if reg.search(item) is not None:

                    return 1

                

        return 0

    

    return temp



# Limit coding to just the most recent version of the app

codes = sql("""

    select *

    from data

    where app_version = 15

""")



# Big-C Codes

# Here I prefer to use the regex lists, since that matches 100%

# with n-coder, where we verified these codes. TODO verify these codes



# The player is taking steps to immediately increase and/or maintain the number of residents of their

# town. Evident when the player attempts to buy a house. Evident when a villager is hungry, then the

# player attempts to buy food. Evident when a villager is hungry, then the player changes the

# use-setting of a corn or milk to be used by the villagers. Does not include buying a farm or dairy

# in response to hunger because the payoff is not immediate.

codes["POPULATION"] = codes["lag_line"].map(regexp(

    "\\bBuy(Fail)?\\(Home[^)]*\\)$",

    "Hungry.*\\bBuy(Fail)?\\(Food[^)]*\\)$",

    "Hungry.*\\bItemUseSelect\\(\\[\\d+, \\d+, (2|5), 1\\]\\)$"

))



# The player is taking steps to increase the total number of floating, non-monetary resources in their

# town. Evident when the player attempts to buy fertilizer. Evident when the player attempts to buy food

# when a villager is not hungry. Evident when the player changes the use-setting of a farm or dairy to be

# used by the villagers. Evident when the player speeds up the game then a farm or dairy produces a

# resource that is set to be used by the villagers. Does not include changing the use-setting on individual

# items, since those items were already there.

codes["RESOURCES"] = codes["lag_line"].map(regexp(

    "\\bBuy(Fail)?\\(Fert[^)]*\\)$", 

    "(?!Hungry).*\\bBuy(Fail)?\\(Food[^)]*\\)$", 

    "\\bTileUseSelect\\(\\[\\d+, \\d+, \\d+, \\d+, \\d+, \\d+\\], [^\\]]*1[^)]*\\)$",

    "Speed\\(Play x(4|16)\\).*(Harvested|Produced)\\([^\\]]*1[^)]*\\)$"

))



# The player is taking steps to eventually increase and/or maintain the amount of in-game money they have. Evident

# when the player attempts to buy a farm, dairy, or road. Evident when the player changes the use-setting

# of a farm, dairy, food, milk, or fertilizer to be sold. Evident when the player speeds up the game then

# a farm or dairy produces a resource that is set to be sold.

codes["BUSINESS"] = codes["lag_line"].map(regexp(

    "\\bBuy(Fail)?\\((Farm|Live|Road)[^)]*\\)$",

    "\\bItemUseSelect\\(\\[\\d+, \\d+, \\d+, 2\\]\\)$",

    "\\bTileUseSelect\\(\\[\\d+, \\d+, \\d+, \\d+, \\d+, \\d+\\], [^\\]]*2[^)]*\\)$",

    "Speed\\(Play x(4|16)\\).*(Harvested|Produced)\\([^\\]]*2[^)]*\\)$"

))



# The player is taking steps to mitigate the destruction of their town’s lake. Evident when the player

# attempts to buy a skim. Evident when the player opens the nutrient overlay then they attempt to buy

# something. Evident by other measures outside of present scope.

codes["AVOIDANCE"] = codes["lag_line"].map(regexp(

    "\\bBuy(Fail)?\\(Skim[^)]*\\)$",

    "OpenMenu\\(Nutrient\\).*\\bBuy(Fail)?\\([^)]*\\)$"

))



# The player is trying and failing, repeatedly. This may support learning through trial and error. Evident

# when the player repeatedly fails to buy things. Evident when the player repeatedly cancels buying things.

# Evident by other measures outside of present scope.

codes["TRIAL_ERROR"] = codes["lag_line"].map(regexp(

    "BuyFail.*BuyFail\\([^)]*\\)$",

    "BuyCancel.*BuyCancel\\([^)]*\\)$"

))



# The system is giving the player feedback that the algae of their lake is not well-managed. Evident when

# there is an algae bloom. Evident when a villager pukes. Evident when the player earns a bloom achievement.

# Evident when the “gross again” tutorial begins.

codes["ALGAE"] = codes["lag_line"].map(regexp(

    "Bloom$",

    "Emote\\(Puke\\)$",

    "Achieve\\(Bloom[^)]*\\)$",

    "gross_again, begin\\)$"

))



# Order-wise small-c codes

codes = sql("""

    with time_ordered as (

        select *, row_number() over (partition by session_id order by time) as ord

        from codes

    )

    select *, (

        select count(*)

        from time_ordered as b

        where a.session_id = b.session_id

        and a.ord > b.ord

        and b.event_custom = 1

        and b.continue = 1

    ) as CONTINUES_SO_FAR

    from time_ordered as a

""")

#     (

#         select count(*)

#         from time_ordered as b

#         where a.session_id = b.session_id

#         and a.ord > b.ord

#         and b.event_custom = 23

#     ) as ENDGAMES,

#     (

#         select count(*)

#         from time_ordered as b

#         where a.session_id = b.session_id

#         and a.ord > b.ord

#         and b.event_custom = 15

#     ) as TUTORIAL_SKIPS

#     from time_ordered as a

# """)



# Unit-wise small-c codes

codes = sql("""

    with units as (

        select session_id,

            count(*) / ((max(seconds) - min(seconds))) as LOG_RATE,

            max(money) as MAX_MONEY,

            max(seconds) as MAX_SECONDS,

            max(CONTINUES_SO_FAR) as CONTINUES,

            max(case

                when http_user_agent like "%(X11%" then 1

                when http_user_agent like "%(Macintosh%" then 1

                when http_user_agent like "%(Windows NT%" then 1

                else 0

            end) as DESKTOP

        from codes

        group by session_id

    )

    select *

    from codes, units

    where codes.session_id = units.session_id

""")



# Line-wise small-c codes (remaining codes by fiat)

codes = sql("""

    select *,

        case

            when event_custom = 17 and 0 <= achievement and achievement <= 3 then 1

            else 0

        end as ACHIEVEPOPULATION,

        case

            when event_custom = 17 and 4 <= achievement and achievement <= 7 then 1

            else 0

        end as ACHIEVEFARM,

        case

            when event_custom = 17 and 8 <= achievement and achievement <= 11 then 1

            else 0

        end as ACHIEVEMONEY,

        case

            when event_custom = 17 and 12 <= achievement and achievement <= 15 then 1

            else 0

        end as ACHIEVEBLOOM,

        case

            when event_custom = 18 then 1

            else 0

        end as DEATH,

        case

            when event_custom = 21 then 1

            else 0

        end as RAINSTOPPED,

        case

            when event_custom = 25 then 1

            else 0

        end as FARMFAIL,

        case

            when event_custom = 27 then 1

            else 0

        end as FARMHARVESTED,

        case

            when event_custom = 28 then 1

            else 0

        end as MILKPRODUCED,

        case

            when event_custom = 29 then 1

            else 0

        end as POOPPRODUCED,

        case

            when event_custom = 31 then 1

            else 0

        end as BIRTH,

        case

            when event_custom = 12 and to_state in (1, "True") then 1

            else 0

        end as OPEN_NUTRIENTS,

        case

            when event_custom = 13 and shop_open in (1, "True") then 1

            else 0

        end as OPEN_SHOP,

        case

            when event_custom = 14 and achievements_open in (1, "True") then 1

            else 0

        end as OPEN_ACHIEVE,

        case

            when event_custom = 16 and cur_speed = 1 then 1

            else 0

        end as PAUSE_GAME,

        case

            when event_custom = 16 and cur_speed = 2 then 1

            else 0

        end as REGULAR_SPEED,

        case

            when event_custom = 16 and cur_speed in (3, 4) then 1

            else 0

        end as INCREASE_SPEED,

        case

            when event_custom in (3, 4, 5) then 1

            else 0

        end as SELECTION,

        case

            when event_custom = 15 then 1

            else 0

        end as SKIP_TUTORIAL

    from codes

""")



# Save that to the disk, so we can analyze with RStudio Cloud + rENA

codes.to_csv("coded.csv", header=True)
# See: https://www.kaggle.com/gregoiredc/survival-analysis-or-nn-predict-age



D = {

    "Everyone": sql("select seconds from codes")["seconds"],

    "PC New Game": sql("select seconds from codes where DESKTOP = 1 and CONTINUES = 0")["seconds"],

    "PC Continue": sql("select seconds from codes where DESKTOP = 1 and CONTINUES > 0")["seconds"],

    "Mobile New Game": sql("select seconds from codes where DESKTOP = 0 and CONTINUES = 0")["seconds"],

    "Mobile Continue": sql("select seconds from codes where DESKTOP = 0 and CONTINUES > 0")["seconds"]

}



# KMF Plot

ax = None

for (label, secs) in D.items():

    ax = lifelines.KaplanMeierFitter().fit(secs, label=label).plot()



# Axis Labels

ax.set_xlim(0, 2700)

ax.set(xlabel="Seconds of Gameplay (up to 45min)", ylabel="Proportion Still Playing (KMF)")

plt.figure(1).savefig("survival.png")
# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html



# KS tests

for i, (labeli, secsi) in enumerate(D.items()):

    if i > 0: # skip Everyone

        for j, (labelj, secsj) in enumerate(D.items()):

            if j > i: # skip pairs we've already compared

                print("#", labeli, "vs", labelj)

                KS = ks_2samp(secsi, secsj)

                print(KS)

                if KS.pvalue >= 0.05:

                    print("FAIL to reject")

                elif KS.statistic <= 1.358 * np.sqrt(float(len(labeli) + len(labelj)) / float(len(labeli) * len(labelj))):

                    print("FAIL to reject")

                else:

                    print("REJECT the null")

                

                print()
def get_examples(desktop, continues, group):

    return sql("""

        with randos as (

            select distinct session_id

            from codes

            where DESKTOP = {0} and CONTINUES {1} 0 and MAX_SECONDS {2} 300

            order by random()

            limit 5

        )

        select session_id, group_concat("    " || timed_line, char(13) || char(10)) as lines

        from (

            select codes.session_id, timed_line,

                row_number() over(partition by codes.session_id order by time) as ord,

                random() % 50 as offset

            from codes, randos

            where codes.session_id = randos.session_id

            and seconds < 300

        )

        where offset <= ord and ord <= 50 + offset

        group by session_id

        order by session_id, ord

    """.format(desktop, continues, group))





examplesDesktopNewShort = get_examples(1, "=", "<")

examplesDesktopNewLong = get_examples(1, "=", ">=")

examplesMobileContShort = get_examples(0, ">", "<")

examplesMobileContLong = get_examples(0, ">", ">=")
def show_examples(f, examples):

    for example in examples.values:

        session_id, lines = example

        f.write("#### {0}\n\n{1}\n\n".format(session_id, lines))



with open("examples.md", "w") as f:

    f.write("# EXAMPLES\n")

    f.write("## DESKTOP, NEW GAME, SHORT\n")

    show_examples(f, examplesDesktopNewShort)

    f.write("## DESKTOP, NEW GAME, LONG\n")

    show_examples(f, examplesDesktopNewLong)

    f.write("## MOBILE, CONTINUE, SHORT\n")

    show_examples(f, examplesMobileContShort)

    f.write("## MOBILE, CONTINUE, LONG\n")

    show_examples(f, examplesMobileContLong)
# D = {

#     "Everyone": sql("select seconds from codes")["seconds"],

#     "No Skips": sql("select seconds from codes where TUTORIAL_SKIPS = 0")["seconds"],

#     "1-4 Skips": sql("select seconds from codes where TUTORIAL_SKIPS in (1, 2, 3, 4)")["seconds"],

#     "5-8 Skips": sql("select seconds from codes where TUTORIAL_SKIPS in (5, 6, 7, 8)")["seconds"],

#     "9-12 Skips": sql("select seconds from codes where TUTORIAL_SKIPS in (9, 10, 11, 12)")["seconds"],

#     "13-16 Skips": sql("select seconds from codes where TUTORIAL_SKIPS in (13, 14, 15, 16)")["seconds"]

# }



# # KMF Plot

# ax = None

# for (label, secs) in D.items():

#     ax = lifelines.KaplanMeierFitter().fit(secs, label=label).plot()



# # Axis Labels

# ax.set_xlim(0, 2700)

# ax.set(xlabel="Seconds of Gameplay (up to 45min)", ylabel="Proportion Still Playing (KMF)")

# plt.figure(1).savefig("survival2.png")
# # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html



# # KS tests

# for i, (labeli, secsi) in enumerate(D.items()):

#     if i > 0: # skip Everyone

#         for j, (labelj, secsj) in enumerate(D.items()):

#             if j > i: # skip pairs we've already compared

#                 print("#", labeli, "vs", labelj)

#                 KS = ks_2samp(secsi, secsj)

#                 print(KS)

#                 if KS.pvalue >= 0.05:

#                     print("FAIL to reject")

#                 elif KS.statistic <= 1.358 * np.sqrt(float(len(labeli) + len(labelj)) / float(len(labeli) * len(labelj))):

#                     print("FAIL to reject")

#                 else:

#                     print("REJECT the null")

                

#                 print()
# TODO ENA, Shaffer says that with enough data ENA *will* show at least as much as the lower level methods; just people responding to each other is a justification for ENA



# TODO do something with the speaker column I just added?
# Small-c codes (codes by fiat)

# Yes, this repeats work already done above in computing the line,

# but it's still quicker to do comparisons right on the columns

# instead of string comparisons on the line text

# codes = sql("""

#     select *,

#         case

#             when event_custom = 12 and to_state in (1, "True") then 1

#             else 0

#         end as OPENNUTRIENTS,

#         case

#             when event_custom = 13 and shop_open in (1, "True") then 1

#             else 0

#         end as OPENSHOP,

#         case

#             when event_custom = 14 and achievements_open in (1, "True") then 1

#             else 0

#         end as OPENACHIEVEMENTS,

#         case

#             when event_custom = 7 then len_buy_hovers

#             when event_custom = 8 then len_buy_hovers

#             else 0

#         end as BUYHOVERS,

#         case

#             when event_custom = 0 then money

#             else 0

#         end as MONEY_OR_ZERO,

#         case

#             when event_custom = 3 then 1

#             when event_custom = 4 then 1

#             when event_custom = 5 then 1

#             else 0

#         end as INSPECT,

#         case

#             when event_custom = 10 then 1

#             when event_custom = 11 then 1

#             else 0

#         end as USESELECT,

#         case

#             when event_custom = 17 and 0 <= achievement and achievement <= 3 then 1

#             else 0

#         end as ACHIEVEPOPULATION,

#         case

#             when event_custom = 17 and achievement = 0 then 1

#             else 0

#         end as ACHIEVEPOPULATION_1,

#         case

#             when event_custom = 17 and achievement = 1 then 1

#             else 0

#         end as ACHIEVEPOPULATION_2,

#         case

#             when event_custom = 17 and achievement = 2 then 1

#             else 0

#         end as ACHIEVEPOPULATION_3,

#         case

#             when event_custom = 17 and achievement = 3 then 1

#             else 0

#         end as ACHIEVEPOPULATION_4,

#         case

#             when event_custom = 17 and 4 <= achievement and achievement <= 7 then 1

#             else 0

#         end as ACHIEVEFARM,

#         case

#             when event_custom = 17 and achievement = 4 then 1

#             else 0

#         end as ACHIEVEFARM_1,

#         case

#             when event_custom = 17 and achievement = 5 then 1

#             else 0

#         end as ACHIEVEFARM_2,

#         case

#             when event_custom = 17 and achievement = 6 then 1

#             else 0

#         end as ACHIEVEFARM_3,

#         case

#             when event_custom = 17 and achievement = 7 then 1

#             else 0

#         end as ACHIEVEFARM_4,

#         case

#             when event_custom = 17 and 8 <= achievement and achievement <= 11 then 1

#             else 0

#         end as ACHIEVEMONEY,

#         case

#             when event_custom = 17 and achievement = 8 then 1

#             else 0

#         end as ACHIEVEMONEY_1,

#         case

#             when event_custom = 17 and achievement = 9 then 1

#             else 0

#         end as ACHIEVEMONEY_2,

#         case

#             when event_custom = 17 and achievement = 10 then 1

#             else 0

#         end as ACHIEVEMONEY_3,

#         case

#             when event_custom = 17 and achievement = 11 then 1

#             else 0

#         end as ACHIEVEMONEY_4,

#         case

#             when event_custom = 17 and 12 <= achievement and achievement <= 15 then 1

#             else 0

#         end as ACHIEVEBLOOM,

#         case

#             when event_custom = 17 and achievement = 12 then 1

#             else 0

#         end as ACHIEVEBLOOM_1,

#         case

#             when event_custom = 17 and achievement = 13 then 1

#             else 0

#         end as ACHIEVEBLOOM_2,

#         case

#             when event_custom = 17 and achievement = 14 then 1

#             else 0

#         end as ACHIEVEBLOOM_3,

#         case

#             when event_custom = 17 and achievement = 15 then 1

#             else 0

#         end as ACHIEVEBLOOM_4

#     from data

#     where app_version = 15

# """)



# # Function for making jittering functions for plotting scatterplots

# def jitter(weight):

#     def temp(x):

#         return x + weight*(np.random.random() - np.random.random() + np.random.random() - np.random.random())

    

#     return temp



# # Helper for plotting scatterplots from the code-and-count model (model zero)

# def help_plot(python, english, jitter_weight=1, xpython="minutes", xenglish="Minute of Gameplay", xjitter_weight=0.1):

#     fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

#     handles, labels = [], []

#     for i, desktop in zero.groupby("desktop"):

#         for j, group in desktop.groupby("continues"):

#             if j >= 2:

#                 continue # skip those rare few who've continued more than once

                

#             k = j + 2*i # a flattened index

#             ax = axs[i, j]



#             # Scatter

#             ax.plot(group[xpython].map(jitter(jitter_weight)),

#                     group[python].map(jitter(xjitter_weight)),

#                     ["x", "x", "o", "o"][k], mfc="none", ms=4,alpha=0.5,

#                     c=["tab:blue", "tab:orange", "tab:blue", "tab:orange"][k],

#                     label=["New iPad", "New PC", "Continue iPad", "Continue PC"][k])

            

#             H, L = ax.get_legend_handles_labels()

#             handles, labels = handles + H, handles + L

    

#             # Trend line

#             z = np.polyfit(group[xpython], group[python], 1)

#             p = np.poly1d(z)

#             ax.plot(group[xpython], p(group[xpython]),"r--")



#     # Labels

#     fig.suptitle("{0} by {1}, Platform, and Continuance".format(english, xenglish))

#     axs[0, 0].set_ylabel("{0}".format(english, xenglish))

#     axs[1, 0].set_xlabel(xenglish)

#     fig.legend(handles, ["New iPad", "New PC", "Cont. iPad", "Cont. PC"],

#                loc="upper center", ncol=4, bbox_to_anchor=(0.475, 0.5575),

#                fancybox=True, shadow=True)

#     fig.savefig("{0}_{1}.png".format(python.lower(), xpython.lower()))



# # TODO

# def help_ftest(python, english, predictors=None, interactions=None):

#     print()

#     print("# " + english)

#     m00 = ols("{0} ~ 1".format(python), data=zero).fit()

#     m01 = ols("{0} ~ desktop + line_rate + minutes + continues".format(python), data=zero).fit()

#     print(m00.summary())

#     print(m01.summary())

#     if predictors is not None:

#         m02 = ols("{0} ~ desktop + line_rate + minutes + continues + {1}".format(python, predictors), data=zero).fit()

#         if interactions is not None:

#             m02 = ols("{0} ~ desktop + line_rate + minutes + continues + ({1})*({2})".format(python, predictors, interactions), data=zero).fit()

            

#         print(m02.summary())

#         print(anova_lm(m01, m02))

#     else:

#         print(anova_lm(m00, m01))



# # Count how many times each code occurred for each player

# zero = sql("""

#     select session_id, continues, desktop, minutes, (count(*) + 1.0)/(max(seconds) - min(seconds) + 1.0) as line_rate,

#         sum(POPULATION) as sum_POPULATION,

#         sum(RESOURCES) as sum_RESOURCES,

#         sum(BUSINESS) as sum_BUSINESS,

#         sum(TOWN) as sum_TOWN,

#         sum(AVOIDANCE) as sum_AVOIDANCE,

#         sum(TRIAL_ERROR) as sum_TRIAL_ERROR,

#         sum(ALGAE) as sum_ALGAE,

#         max(SURVIVED_LONG) as max_SURVIVED_LONG

#     from codes

#     where seconds < 300

#     group by session_id, continues, desktop, minutes

# """)



# # Save to the disk

# zero.to_csv("zero.csv", header=True)



# # Plotting codes and counts

# help_plot("sum_POPULATION", "POPULATION")

# help_plot("sum_RESOURCES", "RESOURCES")

# help_plot("sum_BUSINESS", "BUSINESS")

# help_plot("sum_TOWN", "TOWN")

# help_plot("sum_AVOIDANCE", "AVOIDANCE")

# help_plot("sum_TRIAL_ERROR", "TRIAL_ERROR")

# help_plot("sum_ALGAE", "ALGAE")

# help_plot("max_SURVIVED_LONG", "SURVIVED_LONG")



# # # Hist line_rate

# # fig, ax = plt.subplots()

# # ax.hist(zero["line_rate"], bins=[x/10 for x in range(0, 160, 5)])

# # ax.set_xlabel("Log Row Count")

# # ax.set_ylabel("Avg. Frequency per Second")

# # fig.savefig("zero_line_rate.png")



# # # Hist continues

# # fig, ax = plt.subplots()

# # ax.hist(zero["continues"])

# # ax.set_xlabel("Continue Events")

# # ax.set_ylabel("Frequency")

# # fig.savefig("zero_continues.png")



# # Simple f tests

# help_ftest("sum_POPULATION", "POPULATION")

# help_ftest("sum_RESOURCES", "RESOURCES")

# help_ftest("sum_BUSINESS", "BUSINESS")

# help_ftest("sum_TOWN", "TOWN")

# help_ftest("sum_AVOIDANCE", "AVOIDANCE")

# help_ftest("sum_TRIAL_ERROR", "TRIAL_ERROR")

# help_ftest("sum_ALGAE", "ALGAE")

# help_ftest("max_SURVIVED_LONG", "SURVIVED_LONG")



# # Longer f tests

# help_ftest("max_SURVIVED_LONG", "SURVIVED_LONG",

#            predictors="sum_POPULATION + sum_RESOURCES + sum_BUSINESS + sum_TOWN + sum_AVOIDANCE + sum_TRIAL_ERROR + sum_ALGAE")

# # help_ftest("max_SURVIVED_LONG", "SURVIVED_LONG",

# #            predictors="sum_POPULATION + sum_RESOURCES + sum_BUSINESS + sum_TOWN + sum_AVOIDANCE + sum_TRIAL_ERROR + sum_ALGAE",

# #            interactions="desktop + line_rate + continues")

# # help_ftest("max_SURVIVED_LONG", "SURVIVED_LONG",

# #            predictors="sum_POPULATION + sum_RESOURCES + sum_BUSINESS + sum_TOWN + sum_AVOIDANCE + sum_TRIAL_ERROR + sum_ALGAE",

# #            interactions="sum_POPULATION + sum_RESOURCES + sum_BUSINESS + sum_TOWN + sum_AVOIDANCE + sum_TRIAL_ERROR + sum_ALGAE")



# # # More informed f tests



# # help_ftest("max_MONEY", "Max Money and Nutrient Opens", "sum_OPENNUTRIENTS")

# # help_ftest("sum_INSPECT", "Inspections and Nutrient Opens", "sum_OPENNUTRIENTS")

# # help_ftest("sum_USESELECT", "Use-Selections and Nutrient Opens", "sum_OPENNUTRIENTS")

# # help_ftest("sum_ACHIEVEBLOOM", "Bloom Achievements and Nutrient Opens", "sum_OPENNUTRIENTS")



# # help_ftest("max_MONEY", "Max Money and Shop Opens", "sum_OPENSHOP")

# # help_ftest("sum_INSPECT", "Inspections and Shop Opens", "sum_OPENSHOP")

# # help_ftest("sum_USESELECT", "Use-Selections and Shop Opens", "sum_OPENSHOP")

# # help_ftest("sum_ACHIEVEBLOOM", "Bloom Achievements and Shop Opens", "sum_OPENSHOP")



# # help_ftest("max_MONEY", "Max Money and Achievement Opens", "sum_OPENACHIEVEMENTS")

# # help_ftest("sum_INSPECT", "Inspections and Achievement Opens", "sum_OPENACHIEVEMENTS")

# # help_ftest("sum_USESELECT", "Use-Selections and Achievement Opens", "sum_OPENACHIEVEMENTS")

# # help_ftest("sum_ACHIEVEBLOOM", "Bloom Achievements and Achievement Opens", "sum_OPENACHIEVEMENTS")



# # # More informed plots

# # help_plot("sum_OPENNUTRIENTS", "Nutrient Menu Opens", True, "max_MONEY", "Max Money", False)

# # help_plot("sum_OPENNUTRIENTS", "Nutrient Menu Opens", True, "sum_INSPECT", "Inspections", False)

# # help_plot("sum_OPENSHOP", "Shop Menu Opens", True, "max_MONEY", "Max Money", False)

# # help_plot("sum_OPENSHOP", "Shop Menu Opens", True, "sum_INSPECT", "Inspections", False)

# # help_plot("sum_OPENSHOP", "Shop Menu Opens", True, "sum_USESELECT", "Use-Selections", False)

# # # help_ftest("sum_OPENNUTRIENTS", "Nutrient Opens and Inspections Squared", "sum_INSPECT + np.power(sum_INSPECT, 2)")



# twoway = sql("""

#     select session_id, max(desktop) as desktop, max(SURVIVED_LONG) as SURVIVED_LONG

#     from codes

#     where seconds <= 1500 and continues=0

#     group by session_id

# """)



# print(twoway.groupby(["desktop", "SURVIVED_LONG"]).size())



# twoway = sql("""

#     select session_id, max(desktop) as desktop, max(SURVIVED_LONG) as SURVIVED_LONG

#     from codes

#     where seconds <= 1500 and continues>0

#     group by session_id

# """)



# print(twoway.groupby(["desktop", "SURVIVED_LONG"]).size())
#!/usr/bin/env python3



import pandas as pd

import matplotlib.pyplot as plt





def concat_striker_id(dfp, x):

    ret = dfp[dfp['Player_Id'] == x['Striker_Id']]['Player_Name']

    return ret.values[0]





def concat_match_team(dft, x):

    ret = dft[dft['Team_Id'] == x['Match_Winner_Id']]['Team_Short_Code']

    return ret.values[0]





def concat_man_match(dfp, x):

    ret = dfp[dfp['Player_Id'] == x['Man_Of_The_Match_Id']]['Player_Name']

    return ret.values[0]





def calc(x, mr, mw, mf):

    return (1/3*(x['Batsman_Scored']/mr) +

            (1/3*(x['Wickets_Taken']/mw)) +

            (1/3*(x['Outted']/mf)))





def main(files):

    dfbbb = pd.read_csv(files[0])

    dfp = pd.read_csv(files[1])

    dft = pd.read_csv(files[2])

    dfs = pd.read_csv(files[3])

    dfm = pd.read_csv(files[4]).dropna()

    dfpm = pd.read_csv(files[5])



    # print(dfbbb['Dissimal_Type'].describe())

    # print(dfbbb[dfbbb['Dissimal_Type'] == 'run out'])



    # Who won the most matches

    dfm['Match_Winner_Name'] = dfm.apply(lambda x: concat_match_team(dft, x),

                                         axis=1)



    dfm['Man_of_Match'] = dfm.apply(lambda x: concat_man_match(dfp, x), axis=1)



    mmw = dfm.groupby('Match_Winner_Name')['Match_Id'].count().sort_values(

        ascending=False)

    mmw.plot(kind='barh')

    plt.show()

    # plt.clf()



    # Now plot the most man of the matches

    mmm = dfm.groupby('Man_of_Match')['Match_Id'].count().sort_values(

        ascending=False)[:10]   # Just the top-10

    b = mmm.plot(kind='barh')

    b.set_xlabel('# of MOTM')

    plt.show()

    # plt.clf()



    # Is the MVP also the MOTM?

    dfbbb['Batsman_Scored'] = pd.to_numeric(dfbbb['Batsman_Scored'],

                                            errors='coerce',

                                            downcast='unsigned')



    # Now calculate the total runs scored by players.

    brs = dfbbb[['Striker_Id', 'Batsman_Scored']].groupby('Striker_Id').sum()

    brs = brs.sort_values(ascending=False, by='Batsman_Scored')

    # print(brs.head())



    dfbbb['Player_dissimal_Id'] = pd.to_numeric(dfbbb['Player_dissimal_Id'],

                                                errors='coerce',

                                                downcast='unsigned')

    bwt = dfbbb[dfbbb['Player_dissimal_Id'] > 0][['Match_Id',

                                                  'Bowler_Id']].groupby(

        'Bowler_Id').count()

    bwt = bwt.sort_values(ascending=False, by='Match_Id')

    bwt = bwt.rename(columns={'Match_Id': 'Wickets_Taken'})

    # print(bwt.head())



    # Dismissal type and fielders/wicket keepers

    dfbbb['Fielder_Id'] = pd.to_numeric(dfbbb['Fielder_Id'],

                                        errors='coerce',

                                        downcast='unsigned')

    fv = dfbbb[dfbbb['Fielder_Id'] > 0][['Match_Id',

                                         'Fielder_Id']].groupby(

                                         'Fielder_Id').count()

    fv = fv.sort_values(ascending=False, by='Match_Id')

    fv = fv.rename(columns={'Match_Id': 'Outted'})

    # print(fv.head())



    # Now join the three things

    # brs.set_index('Striker_Id').join(bwt.set_index('Bowler_Id'))

    res = brs.join(bwt).join(fv).reset_index()

    res = res.fillna(value=0)

    res['MVP'] = res.apply(

        lambda x: concat_striker_id(dfp, x), axis=1)

    res = res[['MVP', 'Batsman_Scored', 'Wickets_Taken',

               'Outted']]

    # print(res.head())



    mr = res['Batsman_Scored'].max()

    mw = res['Wickets_Taken'].max()

    mf = res['Outted'].max()

    

    # Weighting the scores for runs scored, wickets taken and outted via fielding/runouts

    res['Final'] = res.apply(lambda x: calc(x, mr, mw, mf), axis=1)



    res = res[['MVP', 'Final']]

    res = res.set_index('MVP').sort_values(ascending=False, by='Final')

    

    # This the MVP -- most valued player, as you can see in the result

    # does not mean getting the man of the match!

    a = res[:10].plot(kind='barh')

    a.set_xlabel('Weighted MVP score')

    plt.show()

    # plt.clf()





main(['../input/Ball_by_Ball.csv', '../input/Player.csv',

      '../input/Team.csv', '../input/Season.csv', '../input/Match.csv',

      '../input/Player_Match.csv'])

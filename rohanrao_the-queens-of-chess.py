## import packages

import numpy as np

import pandas as pd



from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, LinearAxis, Range1d, FactorRange, NumeralTickFormatter

from bokeh.models.tools import HoverTool

from bokeh.palettes import Blues9, Greens9, Oranges7, Reds9, Spectral4

from bokeh.plotting import figure, output_notebook, show

from bokeh.transform import cumsum, dodge



output_notebook()



## setup configuration, constants and parameters

PATH_WGM_GAMES = "../input/chess-games-of-woman-grandmasters/games_wgm.csv"

PATH_FIDE_RATINGS = "../input/chess-fide-ratings"



MAPPING_MONTH = {

    1: "Jan",

    2: "Feb",

    3: "Mar",

    4: "Apr",

    5: "May",

    6: "Jun",

    7: "Jul",

    8: "Aug",

    9: "Sep",

    10: "Oct",

    11: "Nov",

    12: "Dec"

}



MAPPING_WEEKDAY = {

    0: "Mon",

    1: "Tue",

    2: "Wed",

    3: "Thu",

    4: "Fri",

    5: "Sat",

    6: "Sun"

}

## preparing data

df_games = pd.read_csv(PATH_WGM_GAMES, parse_dates = ["end_time"])

df_games = df_games[df_games.rules == "chess"]

df_games["wgm_rating"] = df_games.white_rating

df_games.loc[df_games.wgm_username == df_games.black_username.str.lower(), "wgm_rating"] = df_games[df_games.wgm_username == df_games.black_username.str.lower()].black_rating.values

df_games["wgm_result"] = df_games.white_result

df_games.loc[df_games.wgm_username == df_games.black_username.str.lower(), "wgm_result"] = df_games[df_games.wgm_username == df_games.black_username.str.lower()].black_result.values

df_games["opponent_result"] = df_games.black_result

df_games.loc[df_games.wgm_username == df_games.black_username.str.lower(), "opponent_result"] = df_games[df_games.wgm_username == df_games.black_username.str.lower()].white_result.values

df_games["checkmated"] = (df_games.opponent_result == "checkmated").astype(int)

df_games["year"] = df_games.end_time.dt.year

df_games["month"] = df_games.end_time.dt.month

df_games["date"] = df_games.end_time.dt.date

df_games["weekday"] = df_games.end_time.dt.weekday

df_games["games"] = 1



df_games.head()

print("Games:", df_games.shape[0])

print("First Game:", min(df_games.end_time))

print("Last Game:", max(df_games.end_time))

df_date = df_games.groupby("date")["games"].count().reset_index().sort_values("date")

df_date["cumulative_games"] = df_date.games.cumsum()



source_1 = ColumnDataSource(data = dict(

    date = np.array(df_date.date.values, dtype = np.datetime64),

    games = df_date.games.values,

    cumulative_games = df_date.cumulative_games.values

))



tooltips_11 = [

    ("Date", "@date{%F}"),

    ("Games", "@games")

]



tooltips_12 = [

    ("Date", "@date{%F}"),

    ("Cumulative Games", "@cumulative_games")

]



formatters_1 = {

    "@date": "datetime"

}



v1 = figure(plot_width = 660, plot_height = 300, x_axis_type = "datetime", title = "Games Played")

v1.extra_y_ranges = {"Games": Range1d(start = -32.0, end = 1.1 * max(df_date.games))}



v11 = v1.line("date", "games", source = source_1, width = 2, color = "green", y_range_name = "Games", legend_label = "Games")

v12 = v1.line("date", "cumulative_games", source = source_1, width = 2, color = "blue", legend_label = "Cumulative Games")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_11, formatters = formatters_1, mode = "vline"))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_12, formatters = formatters_1, mode = "vline"))



v1.xaxis.axis_label = "Date"

v1.yaxis.axis_label = "Total Games"

v1.yaxis[0].formatter = NumeralTickFormatter(format = "0a")



v1.add_layout(LinearAxis(y_range_name = "Games", axis_label = "Games"), "right")



v1.legend.location = "top_left"





df_year_games = df_games.groupby("year")["games"].count().reset_index().sort_values("year")

df_year_users = df_games.drop_duplicates(subset = ["year", "wgm_username"]).groupby("year")["games"].count().reset_index().sort_values("year")

df_year_users.rename(columns = {"games": "users"}, inplace = True)

df_year = df_year_games.merge(df_year_users, on = "year")



source_2 = ColumnDataSource(df_year)



tooltips_2 = [

    ("Year", "@year"),

    ("Games", "@games"),

    ("Players", "@users")

]



v2 = figure(plot_width = 660, plot_height = 300, tooltips = tooltips_2, title = "Games and Users by Year")

v2.extra_y_ranges = {"Users": Range1d(start = -5.5, end = 1.1 * max(df_year.users))}



v2.vbar(x = "year", top = "games", source = source_2, width = 0.75, alpha = 0.8, color = "orange", legend_label = "Games Played")

v2.line(x = "year", y = "users", source = source_2, width = 3, y_range_name = "Users", color = "grey", legend_label = "Unique WGM Players")



v2.xaxis.axis_label = "Year"

v2.yaxis.axis_label = "Games"



v2.legend.location = "top_left"



v2.add_layout(LinearAxis(y_range_name = "Users", axis_label = "Users"), "right")





df_month = df_games.groupby("month")["games"].count().reset_index().sort_values("month")

df_month.month = df_month.month.map(MAPPING_MONTH)



source_3 = ColumnDataSource(df_month)



tooltips_3 = [

    ("Month", "@month"),

    ("Games", "@games")

]



v3 = figure(plot_width = 390, plot_height = 300, x_range = df_month.month.values, tooltips = tooltips_3, title = "Games by Month")



v3.vbar(x = "month", top = "games", source = source_3, width = 0.7, alpha = 0.8, color = "maroon")



v3.xaxis.axis_label = "Month"

v3.yaxis.axis_label = "Games"





df_weekday = df_games.groupby("weekday")["games"].count().reset_index().sort_values("weekday")

df_weekday.weekday = df_weekday.weekday.map(MAPPING_WEEKDAY)



source_4 = ColumnDataSource(df_weekday)



tooltips_4 = [

    ("Weekday", "@weekday"),

    ("Games", "@games")

]



v4 = figure(plot_width = 270, plot_height = 300, x_range = df_weekday.weekday.values, tooltips = tooltips_4, title = "Games by Weekday")



v4.vbar(x = "weekday", top = "games", source = source_4, width = 0.7, alpha = 0.8, color = "maroon")



v4.xaxis.axis_label = "Weekday"

v4.yaxis.axis_label = "Games"





show(column(v1, v2, row(v3, v4)))

df_time_class = df_games.groupby("time_class")["games"].count().reset_index()

df_time_class["game_ratio"] = df_time_class.games * 100 / df_time_class.games.sum()

df_time_class["angle"] = df_time_class.games / df_time_class.games.sum() * 2 * np.pi

df_time_class["color"] = Spectral4



tooltips_1 = [

    ("Games Ratio", "@game_ratio{0.0}%")

]



v1 = figure(plot_width = 660, plot_height = 300, tooltips = tooltips_1, title = "Distribution of Game Format")



v1.wedge(

    x = 0,

    y = 1,

    radius = 0.35,

    source = df_time_class,

    start_angle = cumsum("angle", include_zero = True),

    end_angle = cumsum("angle"),

    line_color = "white",

    fill_color = "color",

    legend_field = "time_class"

)



v1.axis.axis_label = None

v1.axis.visible = False

v1.grid.grid_line_color = None





df_bullet = df_games[df_games.time_class == "bullet"].groupby("time_control")["games"].count().reset_index().sort_values("games", ascending = False).head(9)

df_bullet["game_ratio"] = df_bullet.games * 100 / df_bullet.games.sum()

df_bullet["angle"] = df_bullet.games / df_bullet.games.sum() * 2 * np.pi

df_bullet["color"] = Greens9



tooltips_2 = [

    ("Games Ratio", "@game_ratio{0.0}%")

]



v2 = figure(plot_width = 330, plot_height = 250, tooltips = tooltips_2, title = "Time Control of Bullet Games")



v2.wedge(

    x = 0,

    y = 1,

    radius = 0.35,

    source = df_bullet,

    start_angle = cumsum("angle", include_zero = True),

    end_angle = cumsum("angle"),

    line_color = "white",

    fill_color = "color",

    legend_field = "time_control"

)



v2.axis.axis_label = None

v2.axis.visible = False

v2.grid.grid_line_color = None





df_blitz = df_games[df_games.time_class == "blitz"].groupby("time_control")["games"].count().reset_index().sort_values("games", ascending = False).head(9)

df_blitz["game_ratio"] = df_blitz.games * 100 / df_blitz.games.sum()

df_blitz["angle"] = df_blitz.games / df_blitz.games.sum() * 2 * np.pi

df_blitz["color"] = Blues9



tooltips_3 = [

    ("Games Ratio", "@game_ratio{0.0}%")

]



v3 = figure(plot_width = 330, plot_height = 250, tooltips = tooltips_3, title = "Time Control of Blitz Games")



v3.wedge(

    x = 0,

    y = 1,

    radius = 0.35,

    source = df_blitz,

    start_angle = cumsum("angle", include_zero = True),

    end_angle = cumsum("angle"),

    line_color = "white",

    fill_color = "color",

    legend_field = "time_control"

)



v3.axis.axis_label = None

v3.axis.visible = False

v3.grid.grid_line_color = None





df_rapid = df_games[df_games.time_class == "rapid"].groupby("time_control")["games"].count().reset_index().sort_values("games", ascending = False).head(9)

df_rapid["game_ratio"] = df_rapid.games * 100 / df_rapid.games.sum()

df_rapid["angle"] = df_rapid.games / df_rapid.games.sum() * 2 * np.pi

df_rapid["color"] = Reds9



tooltips_4 = [

    ("Games Ratio", "@game_ratio{0.0}%")

]



v4 = figure(plot_width = 330, plot_height = 250, tooltips = tooltips_4, title = "Time Control of Rapid Games")



v4.wedge(

    x = 0,

    y = 1,

    radius = 0.35,

    source = df_rapid,

    start_angle = cumsum("angle", include_zero = True),

    end_angle = cumsum("angle"),

    line_color = "white",

    fill_color = "color",

    legend_field = "time_control"

)



v4.axis.axis_label = None

v4.axis.visible = False

v4.grid.grid_line_color = None





df_daily = df_games[df_games.time_class == "daily"].groupby("time_control")["games"].count().reset_index().sort_values("games", ascending = False).head(9)

df_daily["game_ratio"] = df_daily.games * 100 / df_daily.games.sum()

df_daily["angle"] = df_daily.games / df_daily.games.sum() * 2 * np.pi

df_daily["color"] = Oranges7



tooltips_5 = [

    ("Games Ratio", "@game_ratio{0.0}%")

]



v5 = figure(plot_width = 330, plot_height = 250, tooltips = tooltips_5, title = "Time Control of Daily Games")



v5.wedge(

    x = 0,

    y = 1,

    radius = 0.35,

    source = df_daily,

    start_angle = cumsum("angle", include_zero = True),

    end_angle = cumsum("angle"),

    line_color = "white",

    fill_color = "color",

    legend_field = "time_control"

)



v5.axis.axis_label = None

v5.axis.visible = False

v5.grid.grid_line_color = None



v5.legend.label_text_font_size = "8pt"



show(column(v1, row(v2, v3), row(v4, v5)))

df_users = pd.pivot_table(df_games, index = "wgm_username", columns = "year", values = "games", aggfunc = "sum", fill_value = 0).reset_index()

df_users.columns.name = None

df_users["games"] = df_users.drop("wgm_username", axis = 1).sum(axis = 1)

df_users = df_users.sort_values("games", ascending = False).head(10).sort_values("games")



years = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

games = {

    "user": df_users.wgm_username.values,

    "2009": df_users[2009].values,

    "2010": df_users[2010].values,

    "2011": df_users[2011].values,

    "2012": df_users[2012].values,

    "2013": df_users[2013].values,

    "2014": df_users[2014].values,

    "2015": df_users[2015].values,

    "2016": df_users[2016].values,

    "2017": df_users[2017].values,

    "2018": df_users[2018].values,

    "2019": df_users[2019].values,

    "2020": df_users[2020].values

}



source_1 = ColumnDataSource(games)



v1 = figure(

    plot_width = 660,

    plot_height = 400,

    x_range = (0, 1.1 * max(df_users.games.values)),

    y_range = df_users.wgm_username.values,

    title = "Most Games Played in Lifetime"

)



v1.hbar_stack(years, y = "user", source = source_1, color = tuple(reversed(list(Blues9) + ["#ffffff"] * 3)), height = 0.75,

              legend_label = ["Games in %s" % x for x in years])



v1.xaxis.axis_label = "Games"

v1.yaxis.axis_label = "Player"



v1.legend.location = "bottom_right"





df_users = df_games.groupby(["year", "wgm_username"])["games"].sum().reset_index().sort_values("games", ascending = False)

df_users = df_users.groupby("year").head(3).sort_values(["year", "games"], ascending = [True, False])



factors = list(zip(*[df_users.year.astype(str).values, df_users.wgm_username.values]))



tooltips_2 = [

    ("Games", "$y{0}")

]



v2 = figure(

    plot_width = 660,

    plot_height = 450,

    x_range = FactorRange(*factors),

    tooltips = tooltips_2,

    title = "Most Games Played by Year"

)



v2.vbar(x = factors, top = df_users.games.values, width = 0.75, alpha = 0.5)



v2.xgrid.grid_line_color = None



v2.xaxis.major_label_orientation = np.pi / 2



v2.xaxis.axis_label = "Year and Player"

v2.yaxis.axis_label = "Games"





show(column(v1, v2))

df_white = df_games[df_games.wgm_username == df_games.white_username.str.lower()]

df_black = df_games[df_games.wgm_username == df_games.black_username.str.lower()]



df_white["wins_white"] = (df_white.white_result == "win").astype(int)

df_black["wins_black"] = (df_black.black_result == "win").astype(int)



df_white = df_white.groupby("wgm_username")[["games", "wins_white"]].sum().reset_index().rename(columns = {"games": "games_white"})

df_black = df_black.groupby("wgm_username")[["games", "wins_black"]].sum().reset_index().rename(columns = {"games": "games_black"})



df_users = df_white.merge(df_black, on = "wgm_username", how = "outer")

df_users.fillna(0, inplace = True)

df_users["games"] = df_users.games_white + df_users.games_black

df_users["wins"] = df_users.wins_white + df_users.wins_black

df_users["win_percentage"] = df_users.wins / df_users.games

df_users["win_percentage_white"] = df_users.wins_white / df_users.games_white

df_users["win_percentage_black"] = df_users.wins_black / df_users.games_black



df_users = df_users[df_users.games >= 100].sort_values("win_percentage", ascending = False).head(10).sort_values("win_percentage")

source_1 = ColumnDataSource(data = dict(

    user = df_users.wgm_username.values,

    games = df_users.games.values,

    games_white = df_users.games_white.values,

    games_black = df_users.games_black.values,

    win_percentage = df_users.win_percentage.values * 100,

    win_percentage_white = df_users.win_percentage_white.values * 100,

    win_percentage_black = df_users.win_percentage_black.values * 100,

))



tooltips_11 = [

    ("Games with White", "@games_white"),

    ("Win % with White", "@win_percentage_white{0.0}%"),

]



tooltips_12 = [

    ("Games with Black", "@games_black"),

    ("Win % with Black", "@win_percentage_black{0.0}%"),

]





v1 = figure(

    plot_width = 660,

    plot_height = 450,

    x_range = Range1d(50, 100),

    y_range = df_users.wgm_username.values,

    title = "Highest Win % (At least 100 games played)"

)



v11 = v1.hbar(y = dodge("user", 0.15, range = v1.y_range),

              right = "win_percentage_white", source = source_1, height = 0.2, color = "white", legend_label = "Win % playing White")



v12 = v1.hbar(y = dodge("user", -0.15, range = v1.y_range),

              right = "win_percentage_black", source = source_1, height = 0.2, color = "black", legend_label = "Win % playing Black")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_11))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_12))



v1.background_fill_color = "green"

v1.background_fill_alpha = 0.25



v1.xaxis.axis_label = "Win %"

v1.yaxis.axis_label = "Player"



v1.legend.location = "bottom_right"



show(v1)

df_current = df_games.sort_values("end_time", ascending = False).drop_duplicates("wgm_username")[["wgm_username", "wgm_rating"]].rename(columns = {"wgm_rating": "current_rating"})

df_highest = df_games.groupby("wgm_username")["wgm_rating"].max().reset_index().rename(columns = {"wgm_rating": "highest_rating"})



df_users = df_current.merge(df_highest, on = "wgm_username")



df_users_current = df_users.sort_values("current_rating", ascending = False).head(10)

df_users_highest = df_users.sort_values("highest_rating", ascending = False).head(10)



source_1 = ColumnDataSource(df_users_current)



tooltips_1 = [

    ("Current Rating", "@current_rating")

]



tooltips_2 = [

    ("Highest Rating", "@highest_rating")

]



v1 = figure(plot_width = 660, plot_height = 400, x_range = df_users_current.wgm_username.values, title = "Top Current Ratings")



v11 = v1.line("wgm_username", "current_rating", source = source_1, width = 3, color = "green", alpha = 0.5, legend_label = "Current Rating")

v12 = v1.line("wgm_username", "highest_rating", source = source_1, width = 3, color = "red", alpha = 0.5, legend_label = "Highest Rating")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))



v1.xaxis.major_label_orientation = np.pi / 4



v1.xaxis.axis_label = "Player"

v1.yaxis.axis_label = "Rating"



v1.legend.location = "bottom_left"





source_2 = ColumnDataSource(df_users_highest)



v2 = figure(plot_width = 660, plot_height = 400, x_range = df_users_highest.wgm_username.values, title = "Top Highest Ratings")



v21 = v2.line("wgm_username", "current_rating", source = source_2, width = 3, color = "green", alpha = 0.5, legend_label = "Current Rating")

v22 = v2.line("wgm_username", "highest_rating", source = source_2, width = 3, color = "red", alpha = 0.5, legend_label = "Highest Rating")



v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_1))

v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_2))



v2.xaxis.major_label_orientation = np.pi / 4



v2.xaxis.axis_label = "Player"

v2.yaxis.axis_label = "Rating"



v2.legend.location = "bottom_left"





show(column(v1, v2))

df_fide_2019 = pd.read_csv(f"{PATH_FIDE_RATINGS}/ratings_2019.csv")

df_fide_2020 = pd.read_csv(f"{PATH_FIDE_RATINGS}/ratings_2020.csv")



df_fide = pd.concat([df_fide_2019, df_fide_2020], ignore_index = True)



df_ratings = df_fide[df_fide.fide_id == 13604040]

df_ratings["date"] = pd.to_datetime(df_ratings.year.astype(str) + "-" + df_ratings.month.astype(str) + "-15")



df_online_ratings = df_games[df_games.wgm_username == "meri-arabidze"].groupby("date")["wgm_rating"].mean().reset_index()

df_online_ratings.date = pd.to_datetime(df_online_ratings.date)



df_ratings = df_ratings.merge(df_online_ratings, on = "date", how = "outer").sort_values("date").reset_index(drop = True)

df_ratings.fillna(method = "ffill", inplace = True)

source = ColumnDataSource(data = dict(

    date = np.array(df_ratings.date.values, dtype = np.datetime64),

    classical_rating = df_ratings.rating_standard.values,

    rapid_rating = df_ratings.rating_rapid.values,

    blitz_rating = df_ratings.rating_blitz.values,

    online_rating = df_ratings.wgm_rating.values

))



tooltips = [

    ("Date", "@date{%F}"),

    ("Classical Rating", "@classical_rating{0}"),

    ("Rapid Rating", "@rapid_rating{0}"),

    ("Blitz Rating", "@blitz_rating{0}"),

    ("Online Rating", "@online_rating{0}"),

]



formatters = {

    "@date": "datetime"

}



v = figure(plot_width = 660, plot_height = 400, x_axis_type = "datetime", tooltips = tooltips, title = "FIDE Ratings vs Online Rating (Meri Arabidze)")



v.line("date", "classical_rating", source = source, width = 3, color = "yellow", alpha = 0.5, legend_label = "FIDE Classical Rating")

v.line("date", "rapid_rating", source = source, width = 3, color = "orange", alpha = 0.5, legend_label = "FIDE Rapid Rating")

v.line("date", "blitz_rating", source = source, width = 3, color = "red", alpha = 0.5, legend_label = "FIDE Blitz Rating")

v.line("date", "online_rating", source = source, width = 3, color = "green", alpha = 0.5, legend_label = "Online Chess Rating")



v.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Rating"



v.legend.location = "top_left"



show(v)

!pip install python-chess==0.31.1
import chess.pgn

import io



## reading first game

game = chess.pgn.read_game(io.StringIO(df_games.pgn[0]))

print(game)



## setting up chess board

board = game.board()

board
## visualize first move

board = game.board()

for i, move in enumerate(game.mainline_moves()):

    if i == 0:

        board.push(move)

    else:

        break



board
## visualize nth move (n = 13)

n = 13

board = game.board()

for i, move in enumerate(game.mainline_moves()):

    if i <= n:

        board.push(move)

    else:

        break



board
## visualize end position

board = game.board()

for i, move in enumerate(game.mainline_moves()):

    board.push(move)



board
def get_first_move(pgn):

    """

    Returns the first move of game.

    """

    game = chess.pgn.read_game(io.StringIO(pgn))

    first_move = ""



    for i, move in enumerate(game.mainline_moves()):

        if i == 0:

            first_move = str(move)

        else:

            break



    return first_move



df_games["first_move"] = df_games.pgn.apply(lambda x: get_first_move(x))



df_moves = df_games.groupby("first_move")["games"].count().reset_index()

df_moves["games_percentage"] = round(df_moves.games * 100 / df_moves.games.sum(), 2)

df_moves.sort_values("games", ascending = False).head(4)

df_users = df_games[df_games.wgm_result == "win"].groupby("wgm_username")[["games", "checkmated"]].sum().reset_index()

df_users["checkmate_rate"] = df_users.checkmated / df_users.games

df_users = df_users[df_users.games >= 100].sort_values("checkmate_rate", ascending = False).head(10)



source_1 = ColumnDataSource(data = dict(

    user = df_users.wgm_username.values,

    games_won = df_users.games.values,

    games_checkmated = df_users.checkmated.values,

    checkmate_percentage = df_users.checkmate_rate * 100

))



tooltips_1 = [

    ("Total Games Won", "@games_won")

]



tooltips_2 = [

    ("Games Won by Checkmate", "@games_checkmated")

]



tooltips_3 = [

    ("Checkmate Percentage", "@checkmate_percentage{0.00}%")

]



v1 = figure(

    plot_width = 650,

    plot_height = 400,

    x_range = df_users.wgm_username.values,

    y_range = Range1d(0, 1.1 * max(df_users.games.values)),

    title = "Highest Checkmating Percentage (At least 100 games won)"

)



v1.extra_y_ranges = {"Checkmate Percentage": Range1d(start = 20, end = 40)}



v11 = v1.vbar(x = dodge("user", 0.15, range = v1.x_range), top = "games_won", width = 0.2, source = source_1, color = "blue", legend_label = "Games Won")

v12 = v1.vbar(x = dodge("user", -0.15, range = v1.x_range), top = "games_checkmated", width = 0.2, source = source_1, color = "orange", legend_label = "Games Checkmated")

v13 = v1.line("user", "checkmate_percentage", source = source_1, width = 3, color = "red", y_range_name = "Checkmate Percentage", legend_label = "Checkmate Percentage")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.add_tools(HoverTool(renderers = [v13], tooltips = tooltips_3))



v1.xaxis.major_label_orientation = np.pi / 4



v1.xaxis.axis_label = "Player"

v1.yaxis.axis_label = "Games"

v1.add_layout(LinearAxis(y_range_name = "Checkmate Percentage", axis_label = "Checkmate Percentage"), "right")



v1.legend.location = "top_center"



show(v1)

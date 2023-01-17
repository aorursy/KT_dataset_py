import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 



%matplotlib  inline 
#import dat csv file 

game_data = pd.read_csv("../input/ign.csv", index_col=0)
game_data.head(5)
Computer = ["PC", "Macintosh", "Linux", "Commodore 64/128", "SteamOS"]



Console = ["PlayStation 2", "Xbox 360", "Wii", "PlayStation 3", "PlayStation", "Xbox",

           "GameCube", "Nintendo 64", "Dreamcast", "PlayStation 4", "Xbox One", "Wii U", "Genesis",

           "NES", "TurboGrafx-16", "Super NES", "Sega 32X", "Master System", "Saturn",

           "Atari 2600", "Atari 5200", "TurboGrafx-CD", "Ouya", "NeoGeo","Nintendo 64DD"]



Portable = ["Nintendo DSi", "PlayStation Portable", "Game Boy Advance", "Game Boy Color", "Nintendo 3DS",

            "PlayStation Vita" , "Lynx", "NeoGeo Pocket Color", "Game Boy", "WonderSwan",

            "New Nintendo 3DS", "WonderSwan Color", "dreamcast VMU","Nintendo DS"]



Mobile = ["iPhone", "iPad", "Android", "Windows Phone", "iPod", "Pocket PC", "Windows Surface", "N-Gage"]

Arcade = ["Arcade", "Vectrex"]



def mapping(item):

    if item in Computer:

        return "Computer"

    elif item in Console:

        return "Console"

    elif item in Portable:

        return "Portable"

    elif item in Mobile:

        return "Mobile"

    elif item in Arcade:

        return "Arcade"

    return "Other"



game_data["Game_Type"] = game_data["platform"].map(mapping)
game_type_by_year =  game_data.groupby(['release_year','Game_Type']).size().unstack()

game_type_by_year.reset_index(inplace=True)

game_type_by_year = game_type_by_year.fillna(0)

game_type_by_year.head(5)
game_type_by_year.plot(x = 'release_year' ,  kind = 'bar' , stacked  = True , figsize=(12,9) )




Sony = ["PlayStation","PlayStation 2","PlayStation 3","PlayStation 4","PlayStation Portable","PlayStation Vita"]

Nintendo = ["Wii","Nintendo 64","Wii U","Nintendo DSi","Game Boy Advance","Game Boy Color","Nintendo 3DS"

            ,"Game Boy","New Nintendo 3DS","Nintendo DS","GameCube","NES","Super NES"]

Microsoft = ["Xbox 360","Xbox","Xbox One"] 

Sega = ["Dreamcast","Genesis" , "Sega 32X","Saturn","dreamcast VMU"]

SNK = ["NeoGeo","NeoGeo Pocket Color"]

Bandai = ["WonderSwan","WonderSwan Color"]



def mapping(item):

    if item in Sony:

        return "Sony"

    elif item in Nintendo:

        return "Nintendo"

    elif item in Microsoft:

        return "Microsoft"

    elif item in Sega:

        return "Sega"

    elif item in SNK:

        return "SNK"

    elif item in Bandai:

        return "Bandai"

    return "Other"



game_data["Company"] = game_data["platform"].map(mapping)
game_company_by_year =  game_data.groupby(['release_year','Company']).size().unstack()

game_company_by_year = game_company_by_year.fillna(0)

game_company_by_year.head(5)
game_company_by_year.plot(  kind = 'bar' , stacked  = True , figsize=(12,9) )
game_company_by_type =  game_data.groupby(['Game_Type','Company']).size().unstack()

game_company_by_type = game_company_by_type.fillna(0)

game_company_by_type
game_company_by_type_console = game_data[game_data.Game_Type == 'Console']

game_company_by_type_console = game_company_by_type_console.groupby(['Company','Game_Type']).size().unstack()

game_company_by_type_console.head(5)
game_company_by_type_console.plot.pie(subplots=True,autopct='%0.2f',figsize=(8, 6),fontsize=10)
game_company_sony = game_data[(game_data.Company == 'Sony') & (game_data.Game_Type == 'Console')]

game_company_sony = game_company_sony.groupby(['platform','Game_Type']).size().unstack()

game_company_sony.head(5)
game_company_sony.plot.pie(subplots=True,autopct='%0.2f',figsize=(8, 6),fontsize=10)
game_company_by_type_portable = game_data[game_data.Game_Type == 'Portable']

game_company_by_type_portable = game_company_by_type_portable.groupby(['Company','Game_Type']).size().unstack()

game_company_by_type_portable.head(5)
game_company_by_type_portable.plot.pie(subplots=True,autopct='%0.2f',figsize=(8, 6),fontsize=10)
game_company_nintendo = game_data[(game_data.Company == 'Nintendo') & (game_data.Game_Type == 'Portable')]

game_company_nintendo = game_company_nintendo.groupby(['platform','Game_Type']).size().unstack()

game_company_nintendo.head(5)
game_company_nintendo.plot.pie(subplots=True,autopct='%0.2f',figsize=(8, 6),fontsize=10)
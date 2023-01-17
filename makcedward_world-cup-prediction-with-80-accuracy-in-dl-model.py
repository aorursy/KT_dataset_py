# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
summary_df = pd.read_csv('../input/WorldCups.csv')
summary_df.head()
match_df = pd.read_csv('../input/WorldCupMatches.csv')
match_df.head()
player_df = pd.read_csv('../input/WorldCupPlayers.csv')
player_df.head()
# Players Source :http://www.goal.com/en/news/revealed-every-world-cup-2018-squad-23-man-preliminary-lists/oa0atsduflsv1nsf6oqk576rb
# Coach Source: https://www.fifa.com/worldcup/players/coaches/

worldcup_2018_data = []
worldcup_2018_data.extend([{
    'country': 'russia',
    'group': 'A',
    'coach': 'Cherchesov Stanislav',
    'name': 'Igor Akinfeev, Vladimir Gabulov, Andrey Lunev; Sergei Ignashevich, Mario Fernandes, Vladimir Granat, Fyodor Kudryashov, Andrei Semyonov, Igor Smolnikov, Ilya Kutepov, Aleksandr Yerokhin, Yuri Zhirkov, Daler Kuzyaev, Aleksandr Golovin, Alan Dzagoev, Roman Zobnin, Aleksandr Samedov, Yuri Gazinsky, Anton Miranchuk, Denis Cheryshev, Artyom Dzyuba, Aleksei Miranchuk, Fyodor Smolov'
}, {
    'country': 'saudi_arabia',
    'group': 'A',
    'coach': 'Pizzi Juan Antonio',
    'name': 'Mohammed Al-Owais, Yasser Al-Musailem, Abdullah Al-Mayuf; Mansoor Al-Harbi, Yasser Al-Shahrani, Mohammed Al-Burayk, Motaz Hawsawi, Osama Hawsawi, Ali Al-Bulaihi, Omar Othman; Abdullah Alkhaibari, Abdulmalek Alkhaibri, Abdullah Otayf, Taiseer Al-Jassam, Hussain Al-Moqahwi, Salman Al-Faraj, Mohamed Kanno, Hatan Bahbir, Salem Al-Dawsari, Yahia Al-Shehri, Fahad Al-Muwallad, Mohammad Al-Sahlawi, Muhannad Assiri'
}, {
    'country': 'egypt',
    'group': 'A',
    'coach': 'Cuper Hector',
    'name': 'Essam El Hadary, Mohamed El-Shennawy, Sherif Ekramy; Ahmed Fathi, Abdallah Said, Saad Samir, Ayman Ashraf, Mohamed Abdel-Shafy, Ahmed Hegazi, Ali Gabr, Ahmed Elmohamady, Omar Gaber; Tarek Hamed, Mahmoud Shikabala, Sam Morsy, Mohamed Elneny, Mahmoud Kahraba, Ramadan Sobhi, Trezeguet, Amr Warda; Marwan Mohsen, Mohamed Salah, Mahmoud Elwensh'
}, {
    'country': 'uruguay',
    'group': 'A',
    'coach': 'Tabarez Oscar',
    'name': 'Fernando Muslera, Martin Silva, Martin Campana, Diego Godin, Sebastian Coates, Jose Maria Gimenez, Maximiliano Pereira, Gaston Silva, Martin Caceres, Guillermo Varela, Nahitan Nandez, Lucas Torreira, Matias Vecino, Rodrigo Bentancur, Carlos Sanchez, Giorgian De Arrascaeta, Diego Laxalt, Cristian Rodriguez, Jonathan Urretaviscaya, Cristhian Stuani, Maximiliano Gomez, Edinson Cavani, Luis Suarez'
}, {
    'country': 'portugal',
    'group': 'B',
    'coach': 'Santos Fernando',
    'name': 'Anthony Lopes, Beto, Rui Patricio, Bruno Alves, Cedric Soares, Jose Fonte, Mario Rui, Pepe, Raphael Guerreiro, Ricardo Pereira, Ruben Dias, Adrien Silva, Bruno Fernandes, Joao Mario, Joao Moutinho, Manuel Fernandes, William Carvalho, Andre Silva, Bernardo Silva, Cristiano Ronaldo, Gelson Martins, Goncalo Guedes, Ricardo Quaresma'
}, {
    'country': 'spain',
    'group': 'B',
    'coach': 'Hierro Fernando',
    'name': 'David de Gea, Pepe Reina, Kepa Arrizabalaga; Dani Carvajal, Alvaro Odriozola, Gerard Pique, Sergio Ramos, Nacho, Cesar Azpilicueta, Jordi Alba, Nacho Monreal; Sergio Busquets, Saul Niquez, Koke, Thiago Alcantara, Andres Iniesta, David Silva; Isco, Marcio Asensio, Lucas Vazquez, Iago Aspas, Rodrigo, Diego Costa'
}, {
    'country': 'morocco',
    'group': 'B',
    'coach': 'Renard Herve',
    'name': "Mounir El Kajoui, Yassine Bounou, Ahmad Reda Tagnaouti, Mehdi Benatia, Romain Saiss, Manuel Da Costa, Badr Benoun, Nabil Dirar, Achraf Hakimi, Hamza Mendyl; M'bark Boussoufa, Karim El Ahmadi, Youssef Ait Bennasser, Sofyan Amrabat, Younes Belhanda, Faycal Fajr, Amine Harit; Khalid Boutaib, Aziz Bouhaddouz, Ayoub El Kaabi, Nordin Amrabat, Mehdi Carcela, Hakim Ziyech"
}, {
    'country': 'iran',
    'group': 'B',
    'coach': 'Queiroz Carlos',
    'name': 'Alireza Beiranvand, Rashid Mazaheri, Amir Abedzadeh; Ramin Rezaeian, Mohammad Reza Khanzadeh, Morteza Pouraliganji, Pejman Montazeri, Seyed Majid Hosseini, Milad Mohammadi, Roozbeh Cheshmi; Saeid Ezatolahi, Masoud Shojaei, Saman Ghoddos, Mehdi Torabi, Ashkan Dejagah, Omid Ebrahimi, Ehsan Hajsafi, Vahid Amiri; Alireza Jahanbakhsh, Karim Ansarifard, Mahdi Taremi, Sardar Azmoun, Reza Ghoochannejhad'
}, {
    'country': 'france',
    'group': 'C',
    'coach': 'Deschamps Didier',
    'name': "Alphonse Areola, Hugo Lloris, Steve Mandanda; Lucas Hernandez, Presnel Kimpembe, Benjamin Mendy, Benjamin Pavard, Adil Rami, Djibril Sidibe, Samuel Umtiti, Raphael Varane; N'Golo Kante, Blaise Matuidi, Steven N'Zonzi, Paul Pogba, Corentin Tolisso, Ousmane Dembele, Nabil Fekir; Olivier Giroud, Antoine Griezmann, Thomas Lemar, Kylian Mbappe, Florian Thauvin"
}, {
    'country': 'australia',
    'group': 'C',
    'coach': 'Van Marwur bert',
    'name': 'Brad Jones, Mat Ryan, Danny Vukovic; Aziz Behich, Milos Degenek, Matthew Jurman, James Meredith, Josh Risdon, Trent Sainsbury; Jackson Irvine, Mile Jedinak, Robbie Kruse, Massimo Luongo, Mark Milligan, Aaron Mooy, Tom Rogic; Daniel Arzani, Tim Cahill, Tomi Juric, Mathew Leckie, Andrew Nabbout, Dimitri Petratos, Jamie Maclaren'
}, {
    'country': 'peru',
    'group': 'C',
    'coach': 'Gareca Ricardo',
    'name': 'Carlos Caceda, Jose Carvallo, Pedro Gallese, Luis Advincula, Pedro Aquino, Miguel Araujo, Andre Carrillo, Wilder Cartagena, Aldo Corzo, Christian Cueva, Jefferson Farfan, Edison Flores, Paolo Hurtado, Nilson Loyola, Andy Polo, Christian Ramos, Alberto Rodriguez, Raul Ruidiaz, Anderson Santamaria, Renato Tapia, Miguel Trauco, Yoshimar Yotun, Paolo Guerrero'
}, {
    'country': 'denmark',
    'group': 'C',
    'coach': 'Hareide Age',
    'name': 'Kasper Schmeichel, Jonas Lossl, Frederik Ronow; Simon Kjaer, Andreas Christensen, Mathias Jorgensen, Jannik Vestergaard, Henrik Dalsgaard, Jens Stryger, Jonas Knudsen; William Kvist, Thomas Delaney, Lukas Lerager, Lasse Schone, Christian Eriksen, Michael Krohn-Dehli; Pione Sisto, Martin Braithwaite, Andreas Cornelius, Viktor Fischer, Yussuf Poulsen, Nicolai Jorgensen, Kasper Dolberg'
}, {
    'country': 'argentina',
    'group': 'D',
    'coach': 'Sampaoli Jorge',
    'name': 'Nahuel Guzmán, Willy Caballero, Franco Armani; Gabriel Mercado, Nicolas Otamendi, Federico Fazio, Nicolas Tagliafico, Marcos Rojo, Marcos Acuna, Cristian Ansaldi, Eduardo Salvio; Javier Mascherano, Angel Di Maria, Ever Banega, Lucas Biglia, Manuel Lanzini, Gio Lo Celso, Maximiliano Meza; Lionel Messi, Sergio Aguero, Gonzalo Higuain, Paulo Dybala, Cristian Pavon'
}, {
    'country': 'iceland',
    'group': 'D',
    'coach': 'Hallgrimsson Heimir',
    'name': 'Hannes Thor Halldorsson, Runar Alex Runarsson, Frederik Schram; Kari Arnason, Ari Freyr Skulason, Birkir Mar Saevarsson, Sverrir Ingi Ingason, Hordur Magnusson, Holmar Orn Eyjolfsson, Ragnar Sigurdsson; Johann Berg Gudmundsson, Birkir Bjarnason, Arnor Ingvi Traustason, Emil Hallfredsson, Gylfi Sigurdsson, Olafur Ingi Skulason, Rurik Gislason, Samuel Fridjonsson, Aron Gunnarsson; Alfred Finnbogason, Bjorn Bergmann Sigurdarson, Jon Dadi Bodvarsson, Albert Gudmundsson'
}, {
    'country': 'croatia',
    'group': 'D',
    'coach': 'Dalic Zlatko',
    'name': 'Danijel Subasic, Lovre Kalinic, Dominik Livakovic; Vedran Corluka, Domagoj Vida, Ivan Strinic, Dejan Lovren, Sime Vrsaljko, Josip Pivaric, Tin Jedvaj, Duje Caleta-Car; Luka Modric, Ivan Rakitic, Mateo Kovacic, Milan Badelj, Marcelo Brozovic, Filip Bradaric; Mario Mandzukic, Ivan Perisic, Nikola Kalinic, Andrej Kramaric, Marko Pjaca, Ante Rebic'
}, {
    'country': 'nigeria',
    'group': 'D',
    'coach': 'Rohr Gernot',
    'name': 'Ikechukwu Ezenwa, Daniel Akpeyi, Francis Uzoho; William Troost-Ekong, Leon Balogun, Kenneth Omeruo, Bryan Idowu, Chidozie Awaziem, Abdullahi Shehu, Elderson Echiejile, Tyronne Ebuehi; John Obi Mikel, Ogenyi Onazi, John Ogu, Wilfred Ndidi, Oghenekaro Etebo, Joel Obi; Odion Ighalo, Ahmed Musa, Victor Moses, Alex Iwobi, Kelechi Iheanacho, Simeon Nwankwo'
}, {
    'country': 'brazil',
    'group': 'E',
    'coach': 'Tite',
    'name': ' Alisson, Ederson, Cassio; Danilo, Fagner, Marcelo, Filipe Luis, Thiago Silva, Marquinhos, Miranda, Pedro Geromel; Casemiro, Fernandinho, Paulinho, Fred, Renato Augusto, Philippe Coutinho, Willian, Douglas Costa; Neymar, Taison, Gabriel Jesus, Roberto Firmino'
}, {
    'country': 'switzerland',
    'group': 'E',
    'coach': 'Petkovic Vladimir',
    'name': 'Roman Burki, Yvon Mvogo, Yann Sommer; Manuel Akanji, Johan Djourou, Nico Elvedi, Michael Lang, Stephan Lichtsteiner, Jacques-Francois Moubandje, Ricardo Rodriguez, Fabian Schaer; Valon Behrami, Blerim Dzemaili, Gelson Fernandes, Remo Freuler, Xherdan Shaqiri, Granit Xhaka, Steven Zuber, Denis Zakaria; Josip Drmic, Breel Embolo, Mario Gavranovic, Haris Seferovic'
}, {
    'country': 'costa_rica',
    'group': 'E',
    'coach': 'Ranurez Oscar',
    'name': 'Keylor Navas, Patrick Pemberton, Leonel Moreira, Cristian Gamboa, Ian Smith, Ronald Matarrita, Bryan Oviedo, Oscar Duarte, Giancarlo Gonzalez, Francisco Calvo, Kendall Waston, Johnny Acosta, David Guzman, Yeltsin Tejeda, Celso Borges, Randall Azofeifa, Rodney Wallace, Bryan Ruiz, Daniel Colindres, Christian Bolanos, Johan Venegas, Joel Campbell, Marco Urena'
}, {
    'country': 'serbia',
    'group': 'E',
    'coach': 'Krstajic Mladen',
    'name': ' Vladimir Stojkovic, Predrag Rajkovic, Marko Dmitrovic, Aleksandar Kolarov, Antonio Rukavina, Milan Rodic, Branislav Ivanovic, Uros Spajic, Milos Veljkovic, Dusko Tosic, Nikola Milenkovic; Nemanja Matic, Luka Milivojevic, Marko Grujic, Dusan Tadic, Andrija Zivkovic, Filip Kostic, Nemanja Radonjic, Sergej Milinkovic-Savic, Adem Ljajic; Aleksandar Mitrovic, Aleksandar Prijovic, Luka Jovic'
}, {
    'country': 'germany',
    'group': 'F',
    'coach': 'Low Joachim',
    'name': 'Manuel Neuer, Marc-Andre ter Stegen, Kevin Trapp; Jerome Boateng, Matthias Ginter, Jonas Hector, Mats Hummels, Joshua Kimmich, Marvin Plattenhardt, Antonio Rudiger, Niklas Sule; Julian Brandt, Julian Draxler, Mario Gomez, Leon Goretzka, Ilkay Gundogan, Sami Khedira, Toni Kroos, Thomas Muller, Mesut Ozil, Marco Reus, Sebastian Rudy, Timo Werner'
}, {
    'country': 'mexico',
    'group': 'F',
    'coach': 'Osorio Juan Carlos',
    'name': 'Jesus Corona, Alfredo Talavera, Guillermo Ochoa; Hugo Ayala, Carlos Salcedo, Diego Reyes, Miguel Layun, Hector Moreno, Edson Alvarez; Rafael Marquez, Jonathan dos Santos, Marco Fabian, Giovani dos Santos, Hector Herrera, Andres Guardado; Raul Jimenez, Carlos Vela, Javier Hernandez, Jesus Corona, Oribe Peralta, Javier Aquino, Hirving Lozano'
}, {
    'country': 'sweden',
    'group': 'F',
    'coach': 'Andersson Janne',
    'name': 'Robin Olsen, Karl-Johan Johnsson, Kristoffer Nordfeldt, Mikael Lustig, Victor Lindelof, Andreas Granqvist, Martin Olsson, Ludwig Augustinsson, Filip Helander, Emil Krafth, Pontus Jansson, Sebastian Larsson, Albin Ekdal, Emil Forsberg, Gustav Svensson, Oscar Hiljemark, Viktor Claesson, Marcus Rohden, Jimmy Durmaz, Marcus Berg, John Guidetti, Ola Toivonen, Isaac Kiese Thelin'
}, {
    'country': 'south_korea',
    'group': 'F',
    'coach': 'Shin Taeyong',
    'name': 'Kim Seunggyu, Kim Jinhyeon, Cho Hyeonwoo, Kim Younggwon, Jang Hyunsoo, Jeong Seunghyeon, Yun Yeongseon, Oh Bansuk, Kim Minwoo, Park Jooho, Hong Chul, Go Yohan, Lee Yong, Ki Sungyueng, Jeong Wooyoung, Ju Sejong, Koo Jacheol, Lee Jaesung, Lee Seungwoo, Moon Sunmin, Kim Shinwook, Son Heungmin, Hwang Heechan'
}, {
    'country': 'belgium',
    'group': 'G',
    'coach': 'Martinez Roberto',
    'name': 'Koen Casteels, Thibaut Courtois, Simon Mignolet; Toby Alderweireld, Dedryck Boyata, Vincent Kompany, Thomas Meunier, Thomas Vermaelen, Jan Vertonghen; Nacer Chadli, Kevin De Bruyne, Mousa Dembele, Leander Dendoncker, Marouane Fellaini, Youri Tielemans, Axel Witsel; Michy Batshuayi, Yannick Carrasco, Eden Hazard, Thorgan Hazard, Adnan Januzaj, Romelu Lukaku, Dries Mertens'
}, {
    'country': 'panama',
    'group': 'G',
    'coach': 'Gomez Hernan',
    'name': 'Jose Calderon, Jaime Penedo, Alex Rodríguez; Felipe Baloy, Harold Cummings, Eric Davis, Fidel Escobar, Adolfo Machado, Michael Murillo, Luis Ovalle, Roman Torres; Edgar Barcenas, Armando Cooper, Anibal Godoy, Gabriel Gomez, Valentin Pimentel, Alberto Quintero, Jose Luis Rodriguez; Abdiel Arroyo, Ismael Diaz, Blas Perez, Luis Tejada, Gabriel Torres'
}, {
    'country': 'tunisia',
    'group': 'G',
    'coach': 'Maaloul Nabil',
    'name': 'Farouk Ben Mustapha, Moez Hassen, Aymen Mathlouthi, Rami Bedoui, Yohan Benalouane, Syam Ben Youssef, Dylan Bronn, Oussama Haddadi, Ali Maaloul, Yassine Meriah, Hamdi Nagguez, Anice Badri, Mohamed Amine Ben Amor, Ghaylene Chaalali, Ahmed Khalil, Saifeddine Khaoui, Ferjani Sassi, Ellyes Skhiri, Naim Sliti, Bassem Srarfi, Fakhreddine Ben Youssef, Saber Khalifa, Wahbi Khazri'
}, {
    'country': 'england',
    'group': 'G',
    'coach': 'Southgate Gareth',
    'name': 'Jack Butland, Nick Pope, Jordan Pickford; Fabian Delph, Danny Rose, Eric Dier, Kyle Walker, Kieran Trippier, Trent Alexander-Arnold, Harry Maguire, John Stones, Phil Jones, Gary Cahill; Jordan Henderson, Jesse Lingard, Ruben Loftus-Cheek, Ashley Young, Dele Alli, Raheem Sterling; Harry Kane, Jamie Vardy, Marcus Rashford, Danny Welbeck'
}, {
    'country': 'poland',
    'group': 'H',
    'coach': 'Nawalka Adam',
    'name': 'Bartosz Bialkowski, Lukasz Fabianski, Wojciech Szczesny; Jan Bednarek, Bartosz Bereszynski, Thiago Cionek, Kamil Glik, Artur Jedrzejczyk, Michal Pazdan, Lukasz Piszczek; Jakub Blaszczykowski, Jacek Goralski, Kamil Goricki, Grzegorz Krychowiak, Slawomir Peszko, Maciej Rybus, Piotr Zielinski, Rafal Kurzawa, Karol Linetty; Dawid Kownacki, Robert Lewandowski, Arkadiusz Milik, Lukasz Teodorczyk'
}, {
    'country': 'senegal',
    'group': 'H',
    'coach': 'Cisse Aliou',
    'name': 'Abdoulaye Diallo, Khadim Ndiaye, Alfred Gomis, Lamine Gassama, Moussa Wague, Saliou Ciss, Youssouf Sabaly, Kalidou Koulibaly, Salif Sane, Cheikhou Kouyate, Kara Mbodji, Idrisa Gana Gueye, Cheikh Ndoye, Alfred Ndiaye, Pape Alioune Ndiaye, Moussa Sow, Moussa Konate, Diafra Sakho, Sadio Mane, Ismaila Sarr, Mame Biram Diouf, Mbaye Niang, Diao Keita Balde'
}, {
    'country': 'colombia',
    'group': 'H',
    'coach': 'Pekerman Jose',
    'name': 'David Ospina, Camilo Vargas, Jose Fernando Cuadrado; Cristian Zapata, Davinson Sanchez, Santiago Arias, Oscar Murillo, Frank Fabra, Johan Mojica, Yerry Mina; Wilmar Barrios, Carlos Sanchez, Jefferson Lerma, Jose Izquierdo, James Rodriguez, Abel Aguilar, Juan Fernando Quintero, Mateus Uribe, Juan Guillermo Cuadrado; Radamel Falcao Garcia, Miguel Borja, Carlos Bacca, Luis Fernando Muriel'
}, {
    'country': 'japan',
    'group': 'H',
    'coach': 'Nishino Akira',
    'name': 'Eiji Kawashima, Masaaki Higashiguchi, Kosuke Nakamura, Yuto Nagatomo, Tomoaki Makino, Maya Yoshida, Hiroki Sakai, Gotoku Sakai, Gen Shoji, Wataru Endo, Naomichi Ueda, Makoto Hasebe, Keisuke Honda, Takashi Inui, Shinji Kagawa, Hotaru Yamaguchi, Genki Haraguchi, Takashi Usami, Gaku Shibasaki, Ryota Oshima, Shinji Okazaki, Yuya Osako, Yoshinori Muto'
}])

worldcup_2018_data
worldcup_2018_df = pd.DataFrame(worldcup_2018_data)

def clean_merge(row):
    name = row['name'].replace(';', ',').strip()
    names = name.split(',')
    names.append(row['coach'])
    names = sorted(names)
    return ' '.join(names)

worldcup_2018_df['participants'] = worldcup_2018_df.apply(clean_merge, axis=1)

worldcup_2018_df['label'] = 0
worldcup_2018_df.head()
def country_name_code_mapping(df):
    code2name = {}
    name2code = {}
    name2pos = {}
    pos2name = {}

    working_df = df[['Home Team Name', 'Home Team Initials']].drop_duplicates()

    for i, row in working_df.iterrows():
        code2name[row['Home Team Initials']] = row['Home Team Name']
        name2code[row['Home Team Name']] = row['Home Team Initials']
        
    for i, name in enumerate(name2code):
        name2pos[name] = i
        pos2name[i] = name
    
    print('Name to Code Sample')
    for x in name2code:
        print(x, name2code[x])
        break

    print('Code to Name Sample')
    for x in code2name:
        print(x, code2name[x])
        break
        
    print('Name to Position Sample')
    for x in name2pos:
        print(x, name2pos[x])
        break
        
    print('Position to Name Sample')
    for x in pos2name:
        print(x, pos2name[x])
        break
        
    return code2name, name2code, name2pos, pos2name

code2name, name2code, name2pos, pos2name = country_name_code_mapping(match_df)
def join_participants(match_id, team):
    working_df = player_df[
        (player_df['MatchID'] == match_id)
        & (player_df['Team Initials'] == team)
    ]
    coachs = working_df['Coach Name'].unique().tolist()
    players = working_df['Player Name'].unique().tolist()
    
    return coachs + players

match_df['home_participants'] = match_df.apply(lambda x: join_participants(x['MatchID'], x['Home Team Initials']), axis=1)
match_df['away_participants'] = match_df.apply(lambda x: join_participants(x['MatchID'], x['Away Team Initials']), axis=1)
match_df.head()
# code2name, name2code, name2pos, pos2name

# results = []

def _build_record(year, team_name):
    list_of_home_participants = match_df[
        (match_df['Year'] == year)
        & (match_df['Home Team Name'] == team_name)
    ]['home_participants'].tolist()
    
    list_of_away_participants = match_df[
        (match_df['Year'] == year)
        & (match_df['Away Team Name'] == team_name)
    ]['away_participants'].tolist()
    
    participants = []
    for ps in list_of_home_participants + list_of_away_participants:
        participants.extend(ps)
    participants = sorted(list(set(participants)))
    
    return ' '.join(participants)

def get_non_first_fouth_team(year, positive_team_names):
    home_names = match_df[match_df['Year'] == year]['Home Team Name'].unique().tolist()
    away_names = match_df[match_df['Year'] == year]['Away Team Name'].unique().tolist()
    non_winner_names = list(set(home_names + away_names))
    for name in positive_team_names:
        non_winner_names.remove(name)
        
    return non_winner_names

def build_negative(year, positive_team_names):
    non_winner_names = get_non_first_fouth_team(year, positive_team_names)
    
    results = []
    for name in non_winner_names:
        results.append({
            'label': 0,
            'name': _build_record(year, name)
        })
    
    return results

trainin_data = []
for i, row in summary_df.iterrows():
    # positve sample
    trainin_data.append({
        'label': 1,
        'name': _build_record(row['Year'], row['Winner'])
    })
    trainin_data.append({
        'label': 2,
        'name': _build_record(row['Year'], row['Runners-Up'])
    })
    trainin_data.append({
        'label': 3,
        'name': _build_record(row['Year'], row['Third'])
    })
    trainin_data.append({
        'label': 4,
        'name': _build_record(row['Year'], row['Fourth'])
    })
    
    winner_names = [row['Winner'], row['Runners-Up'], row['Third'], row['Fourth']]
    
    # negative sample
    results = build_negative(row['Year'], winner_names)
    trainin_data.extend(results)
    
training_df = pd.DataFrame(trainin_data)

print('Number of Training Record: %d' % len(training_df))
training_df.head()
from nltk.tokenize import sent_tokenize

class CharCNN:
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
    
    def __init__(self, max_len_of_sentence, max_num_of_setnence, verbose=10):
        self.max_len_of_sentence = max_len_of_sentence
        self.max_num_of_setnence = max_num_of_setnence
        self.verbose = verbose
        
        self.num_of_char = 0
        self.num_of_label = 0
        self.unknown_label = ''
        
    def build_char_dictionary(self, char_dict=None, unknown_label='UNK'):
        """
            Define possbile char set. Using "UNK" if character does not exist in this set
        """ 
        
        if char_dict is None:
            char_dict = self.CHAR_DICT
            
        self.unknown_label = unknown_label

        chars = []

        for c in char_dict:
            chars.append(c)

        chars = list(set(chars))
        
        chars.insert(0, unknown_label)

        self.num_of_char = len(chars)
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        
        if self.verbose > 5:
            print('Totoal number of chars:', self.num_of_char)

            print('First 3 char_indices sample:', {k: self.char_indices[k] for k in list(self.char_indices)[:3]})
            print('First 3 indices_char sample:', {k: self.indices_char[k] for k in list(self.indices_char)[:3]})
            

        return self.char_indices, self.indices_char, self.num_of_char
    
    def convert_labels(self, labels):
        """
            Convert label to numeric
        """
        self.label2indexes = dict((l, i) for i, l in enumerate(labels))
        self.index2labels = dict((i, l) for i, l in enumerate(labels))

        if self.verbose > 5:
            print('Label to Index: ', self.label2indexes)
            print('Index to Label: ', self.index2labels)
            
        self.num_of_label = len(self.label2indexes)

        return self.label2indexes, self.index2labels
    
    def _transform_raw_data(self, df, x_col, y_col, label2indexes=None, sample_size=None):
        """
            ##### Transform raw data to list
        """
        
        x = []
        y = []

        actual_max_sentence = 0
        
        if sample_size is None:
            sample_size = len(df)

        for i, row in df.head(sample_size).iterrows():
            x_data = row[x_col]
            y_data = row[y_col]

            sentences = sent_tokenize(x_data)
            x.append(sentences)

            if len(sentences) > actual_max_sentence:
                actual_max_sentence = len(sentences)

            y.append(label2indexes[y_data])

        if self.verbose > 5:
            print('Number of news: %d' % (len(x)))
            print('Actual max sentence: %d' % actual_max_sentence)

        return x, y
    
    def _transform_training_data(self, x_raw, y_raw, max_len_of_sentence=None, max_num_of_setnence=None):
        """
            ##### Transform preorcessed data to numpy
        """
        unknown_value = self.char_indices[self.unknown_label]
        
        x = np.ones((len(x_raw), max_num_of_setnence, max_len_of_sentence), dtype=np.int64) * unknown_value
        y = np.array(y_raw)
        
        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_setnence is None:
            max_num_of_setnence = self.max_num_of_setnence

        for i, doc in enumerate(x_raw):
            for j, sentence in enumerate(doc):
                if j < max_num_of_setnence:
                    for t, char in enumerate(sentence[-max_len_of_sentence:]):
                        if char not in self.char_indices:
                            x[i, j, (max_len_of_sentence-1-t)] = self.char_indices['UNK']
                        else:
                            x[i, j, (max_len_of_sentence-1-t)] = self.char_indices[char]

        return x, y

    def _build_character_block(self, block, dropout=0.3, filters=[64, 100], kernel_size=[3, 3], 
                         pool_size=[2, 2], padding='valid', activation='relu', 
                         kernel_initializer='glorot_normal'):
        
        for i in range(len(filters)):
            block = Conv1D(
                filters=filters[i], kernel_size=kernel_size[i],
                padding=padding, activation=activation, kernel_initializer=kernel_initializer)(block)

        block = Dropout(dropout)(block)
        block = MaxPooling1D(pool_size=pool_size[i])(block)

        block = GlobalMaxPool1D()(block)
        block = Dense(128, activation='relu')(block)
        return block
    
    def _build_sentence_block(self, max_len_of_sentence, max_num_of_setnence, 
                              char_dimension=16,
                              filters=[[3, 5, 7], [200, 300, 300], [300, 400, 400]], 
#                               filters=[[100, 200, 200], [200, 300, 300], [300, 400, 400]], 
                              kernel_sizes=[[4, 3, 3], [5, 3, 3], [6, 3, 3]], 
                              pool_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                              dropout=0.4):
        
        sent_input = Input(shape=(max_len_of_sentence, ), dtype='int64')
        embedded = Embedding(self.num_of_char, char_dimension, input_length=max_len_of_sentence)(sent_input)
        
        blocks = []
        for i, filter_layers in enumerate(filters):
            blocks.append(
                self._build_character_block(
                    block=embedded, filters=filters[i], kernel_size=kernel_sizes[i], pool_size=pool_sizes[i])
            )

        sent_output = concatenate(blocks, axis=-1)
        sent_output = Dropout(dropout)(sent_output)
        sent_encoder = Model(inputs=sent_input, outputs=sent_output)

        return sent_encoder
    
    def _build_document_block(self, sent_encoder, max_len_of_sentence, max_num_of_setnence, 
                             num_of_label, dropout=0.3, 
                             loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        doc_input = Input(shape=(max_num_of_setnence, max_len_of_sentence), dtype='int64')
        doc_output = TimeDistributed(sent_encoder)(doc_input)

        doc_output = Bidirectional(LSTM(128, return_sequences=False, dropout=dropout))(doc_output)

        doc_output = Dropout(dropout)(doc_output)
        doc_output = Dense(128, activation='relu')(doc_output)
        doc_output = Dropout(dropout)(doc_output)
        doc_output = Dense(num_of_label, activation='sigmoid')(doc_output)

        doc_encoder = Model(inputs=doc_input, outputs=doc_output)
        doc_encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return doc_encoder
    
    def preporcess(self, labels, char_dict=None, unknown_label='UNK'):
        if self.verbose > 3:
            print('-----> Stage: preprocess')
            
        self.build_char_dictionary(char_dict, unknown_label)
        self.convert_labels(labels)
    
    def process(self, df, x_col, y_col, 
                max_len_of_sentence=None, max_num_of_setnence=None, label2indexes=None, sample_size=None):
        if self.verbose > 3:
            print('-----> Stage: process')
            
        if sample_size is None:
            sample_size = 1000
        if label2indexes is None:
            if self.label2indexes is None:
                raise Exception('Does not initalize label2indexes. Please invoke preprocess step first')
            label2indexes = self.label2indexes
        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_setnence is None:
            max_num_of_setnence = self.max_num_of_setnence

        x_preprocess, y_preprocess = self._transform_raw_data(
            df=df, x_col=x_col, y_col=y_col, label2indexes=label2indexes)
        
        x_preprocess, y_preprocess = self._transform_training_data(
            x_raw=x_preprocess, y_raw=y_preprocess,
            max_len_of_sentence=max_len_of_sentence, max_num_of_setnence=max_num_of_setnence)
        
        if self.verbose > 5:
            print('Shape: ', x_preprocess.shape, y_preprocess.shape)

        return x_preprocess, y_preprocess
    
    def build_model(self, char_dimension=16, display_summary=False, display_architecture=False, 
                    loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        if self.verbose > 3:
            print('-----> Stage: build model')
            
        sent_encoder = self._build_sentence_block(
            char_dimension=char_dimension,
            max_len_of_sentence=self.max_len_of_sentence, max_num_of_setnence=self.max_num_of_setnence)
                
        doc_encoder = self._build_document_block(
            sent_encoder=sent_encoder, num_of_label=self.num_of_label,
            max_len_of_sentence=self.max_len_of_sentence, max_num_of_setnence=self.max_num_of_setnence, 
            loss=loss, optimizer=optimizer, metrics=metrics)
        
        if display_architecture:
            print('Sentence Architecture')
            IPython.display.display(SVG(model_to_dot(sent_encoder).create(prog='dot', format='svg')))
            print()
            print('Document Architecture')
            IPython.display.display(SVG(model_to_dot(doc_encoder).create(prog='dot', format='svg')))
        
        if display_summary:
            print(doc_encoder.summary())
            
        
        self.model = {
            'sent_encoder': sent_encoder,
            'doc_encoder': doc_encoder
        }
        
        return doc_encoder
    
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=1, shuffle=True):
        if self.verbose > 3:
            print('-----> Stage: train model')
            
        self.get_model().fit(
            x_train, y_train, validation_data=(x_test, y_test), 
            batch_size=batch_size, epochs=epochs, shuffle=shuffle)
        
#         return self.model['doc_encoder']

    def predict(self, x, model=None, return_prob=False):
        if self.verbose > 3:
            print('-----> Stage: predict')
            
        if model is None:
            model = self.get_model()
            
        if return_prob:
            return model.predict(x_test)
        
        return model.predict(x_test).argmax(axis=-1)
    
    def get_model(self):
        return self.model['doc_encoder']
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(training_df, test_size=0.2)
char_cnn = CharCNN(max_len_of_sentence=256, max_num_of_setnence=1)
char_cnn.preporcess(labels=training_df['label'].unique())
x_train, y_train = char_cnn.process(
    df=train_df, x_col='name', y_col='label')
x_test, y_test = char_cnn.process(
    df=test_df, x_col='name', y_col='label')
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Bidirectional
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, Embedding
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

char_cnn.build_model()
char_cnn.train(x_train, y_train, x_test, y_test, batch_size=32, epochs=10)
# Passing dummpy label and getting dummpy y_real just because try to reuse defined function to convert input
x_real, y_real = char_cnn.process(
    df=worldcup_2018_df, x_col='participants', y_col='label')
char_cnn.predict(x_real)
import random

import itertools

import pandas as pd

import numpy as np

import seaborn as sns

from io import StringIO

from hallucinate import Experiment, EstimatorConfig, Kaggle

sns.set(font_scale=1.2)

sns.set_style('whitegrid')

%matplotlib inline
# I have to appologise for the spam in advance here

# This has everything to do with the "reverse engineering" part, we won't cheat and submit

# these to get a 1.0 public LB. Please don't do it yourself as all you can manage is be super

# annoying and spam the public LB.

# I've prepared this from the existing public data, I think I've got the complete data set from

# Jake VanderPlas's resources somewhere and merged it with Kaggle's test set

# Sorry for the spam cell but the only other way to get it (as Kaggle kernels can't see

# the outside Internet) was to include it in the library. That may happen one day.



test_complete_csv_string = '''PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,Survived

892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q,0

893,3,"Wilkes, Mrs. James (Ellen Needs)",female,47.0,1,0,363272,7.0,,S,1

894,2,"Myles, Mr. Thomas Francis",male,62.0,0,0,240276,9.6875,,Q,0

895,3,"Wirz, Mr. Albert",male,27.0,0,0,315154,8.6625,,S,0

896,3,"Hirvonen, Mrs. Alexander (Helga E Lindqvist)",female,22.0,1,1,3101298,12.2875,,S,1

897,3,"Svensson, Mr. Johan Cervin",male,14.0,0,0,7538,9.225,,S,1

898,3,"Connolly, Miss. Kate",female,30.0,0,0,330972,7.6292,,Q,0

899,2,"Caldwell, Mr. Albert Francis",male,26.0,1,1,248738,29.0,,S,1

900,3,"Abrahim, Mrs. Joseph (Sophie Halaut Easu)",female,18.0,0,0,2657,7.2292,,C,1

901,3,"Davies, Mr. John Samuel",male,21.0,2,0,A/4 48871,24.15,,S,0

902,3,"Ilieff, Mr. Ylio",male,,0,0,349220,7.8958,,S,0

903,1,"Jones, Mr. Charles Cresson",male,46.0,0,0,694,26.0,,S,0

904,1,"Snyder, Mrs. John Pillsbury (Nelle Stevenson)",female,23.0,1,0,21228,82.2667,B45,S,1

905,2,"Howard, Mr. Benjamin",male,63.0,1,0,24065,26.0,,S,0

906,1,"Chaffee, Mrs. Herbert Fuller (Carrie Constance Toogood)",female,47.0,1,0,W.E.P. 5734,61.175,E31,S,1

907,2,"del Carlo, Mrs. Sebastiano (Argenia Genovesi)",female,24.0,1,0,SC/PARIS 2167,27.7208,,C,1

908,2,"Keane, Mr. Daniel",male,35.0,0,0,233734,12.35,,Q,0

909,3,"Assaf, Mr. Gerios",male,21.0,0,0,2692,7.225,,C,0

910,3,"Ilmakangas, Miss. Ida Livija",female,27.0,1,0,STON/O2. 3101270,7.925,,S,0

911,3,"Assaf Khalil, Mrs. Mariana (""Miriam"")",female,45.0,0,0,2696,7.225,,C,1

912,1,"Rothschild, Mr. Martin",male,55.0,1,0,PC 17603,59.4,,C,0

913,3,"Olsen, Master. Artur Karl",male,9.0,0,1,C 17368,3.1708,,S,1

914,1,"Flegenheim, Mrs. Alfred (Antoinette)",female,,0,0,PC 17598,31.6833,,S,1

915,1,"Williams, Mr. Richard Norris II",male,21.0,0,1,PC 17597,61.3792,,C,1

916,1,"Ryerson, Mrs. Arthur Larned (Emily Maria Borie)",female,48.0,1,3,PC 17608,262.375,B57 B59 B63 B66,C,1

917,3,"Robins, Mr. Alexander A",male,50.0,1,0,A/5. 3337,14.5,,S,0

918,1,"Ostby, Miss. Helene Ragnhild",female,22.0,0,1,113509,61.9792,B36,C,1

919,3,"Daher, Mr. Shedid",male,22.5,0,0,2698,7.225,,C,0

920,1,"Brady, Mr. John Bertram",male,41.0,0,0,113054,30.5,A21,S,0

921,3,"Samaan, Mr. Elias",male,,2,0,2662,21.6792,,C,0

922,2,"Louch, Mr. Charles Alexander",male,50.0,1,0,SC/AH 3085,26.0,,S,0

923,2,"Jefferys, Mr. Clifford Thomas",male,24.0,2,0,C.A. 31029,31.5,,S,0

924,3,"Dean, Mrs. Bertram (Eva Georgetta Light)",female,33.0,1,2,C.A. 2315,20.575,,S,1

925,3,"Johnston, Mrs. Andrew G (Elizabeth ""Lily"" Watson)",female,,1,2,W./C. 6607,23.45,,S,0

926,1,"Mock, Mr. Philipp Edmund",male,30.0,1,0,13236,57.75,C78,C,1

927,3,"Katavelas, Mr. Vassilios (""Catavelas Vassilios"")",male,18.5,0,0,2682,7.2292,,C,0

928,3,"Roth, Miss. Sarah A",female,,0,0,342712,8.05,,S,1

929,3,"Cacic, Miss. Manda",female,21.0,0,0,315087,8.6625,,S,0

930,3,"Sap, Mr. Julius",male,25.0,0,0,345768,9.5,,S,1

931,3,"Hee, Mr. Ling",male,,0,0,1601,56.4958,,S,1

932,3,"Karun, Mr. Franz",male,39.0,0,1,349256,13.4167,,C,1

933,1,"Franklin, Mr. Thomas Parham",male,,0,0,113778,26.55,D34,S,0

934,3,"Goldsmith, Mr. Nathan",male,41.0,0,0,SOTON/O.Q. 3101263,7.85,,S,0

935,2,"Corbett, Mrs. Walter H (Irene Colvin)",female,30.0,0,0,237249,13.0,,S,0

936,1,"Kimball, Mrs. Edwin Nelson Jr (Gertrude Parsons)",female,45.0,1,0,11753,52.5542,D19,S,1

937,3,"Peltomaki, Mr. Nikolai Johannes",male,25.0,0,0,STON/O 2. 3101291,7.925,,S,0

938,1,"Chevre, Mr. Paul Romaine",male,45.0,0,0,PC 17594,29.7,A9,C,1

939,3,"Shaughnessy, Mr. Patrick",male,,0,0,370374,7.75,,Q,0

940,1,"Bucknell, Mrs. William Robert (Emma Eliza Ward)",female,60.0,0,0,11813,76.2917,D15,C,1

941,3,"Coutts, Mrs. William (Winnie ""Minnie"" Treanor)",female,36.0,0,2,C.A. 37671,15.9,,S,1

942,1,"Smith, Mr. Lucien Philip",male,24.0,1,0,13695,60.0,C31,S,0

943,2,"Pulbaum, Mr. Franz",male,27.0,0,0,SC/PARIS 2168,15.0333,,C,0

944,2,"Hocking, Miss. Ellen ""Nellie""",female,20.0,2,1,29105,23.0,,S,1

945,1,"Fortune, Miss. Ethel Flora",female,28.0,3,2,19950,263.0,C23 C25 C27,S,1

946,2,"Mangiavacchi, Mr. Serafino Emilio",male,,0,0,SC/A.3 2861,15.5792,,C,0

947,3,"Rice, Master. Albert",male,10.0,4,1,382652,29.125,,Q,0

948,3,"Cor, Mr. Bartol",male,35.0,0,0,349230,7.8958,,S,0

949,3,"Abelseth, Mr. Olaus Jorgensen",male,25.0,0,0,348122,7.65,F G63,S,1

950,3,"Davison, Mr. Thomas Henry",male,,1,0,386525,16.1,,S,0

951,1,"Chaudanson, Miss. Victorine",female,36.0,0,0,PC 17608,262.375,B61,C,1

952,3,"Dika, Mr. Mirko",male,17.0,0,0,349232,7.8958,,S,0

953,2,"McCrae, Mr. Arthur Gordon",male,32.0,0,0,237216,13.5,,S,0

954,3,"Bjorklund, Mr. Ernst Herbert",male,18.0,0,0,347090,7.75,,S,0

955,3,"Bradley, Miss. Bridget Delia",female,22.0,0,0,334914,7.725,,Q,1

956,1,"Ryerson, Master. John Borie",male,13.0,2,2,PC 17608,262.375,B57 B59 B63 B66,C,1

957,2,"Corey, Mrs. Percy C (Mary Phyllis Elizabeth Miller)",female,,0,0,F.C.C. 13534,21.0,,S,0

958,3,"Burns, Miss. Mary Delia",female,18.0,0,0,330963,7.8792,,Q,0

959,1,"Moore, Mr. Clarence Bloomfield",male,47.0,0,0,113796,42.4,,S,0

960,1,"Tucker, Mr. Gilbert Milligan Jr",male,31.0,0,0,2543,28.5375,C53,C,1

961,1,"Fortune, Mrs. Mark (Mary McDougald)",female,60.0,1,4,19950,263.0,C23 C25 C27,S,1

962,3,"Mulvihill, Miss. Bertha E",female,24.0,0,0,382653,7.75,,Q,1

963,3,"Minkoff, Mr. Lazar",male,21.0,0,0,349211,7.8958,,S,0

964,3,"Nieminen, Miss. Manta Josefina",female,29.0,0,0,3101297,7.925,,S,0

965,1,"Ovies y Rodriguez, Mr. Servando",male,28.5,0,0,PC 17562,27.7208,D43,C,0

966,1,"Geiger, Miss. Amalie",female,35.0,0,0,113503,211.5,C130,C,1

967,1,"Keeping, Mr. Edwin",male,32.5,0,0,113503,211.5,C132,C,0

968,3,"Miles, Mr. Frank",male,,0,0,359306,8.05,,S,0

969,1,"Cornell, Mrs. Robert Clifford (Malvina Helen Lamson)",female,55.0,2,0,11770,25.7,C101,S,1

970,2,"Aldworth, Mr. Charles Augustus",male,30.0,0,0,248744,13.0,,S,0

971,3,"Doyle, Miss. Elizabeth",female,24.0,0,0,368702,7.75,,Q,0

972,3,"Boulos, Master. Akar",male,6.0,1,1,2678,15.2458,,C,0

973,1,"Straus, Mr. Isidor",male,67.0,1,0,PC 17483,221.7792,C55 C57,S,0

974,1,"Case, Mr. Howard Brown",male,49.0,0,0,19924,26.0,,S,0

975,3,"Demetri, Mr. Marinko",male,,0,0,349238,7.8958,,S,0

976,2,"Lamb, Mr. John Joseph",male,,0,0,240261,10.7083,,Q,0

977,3,"Khalil, Mr. Betros",male,,1,0,2660,14.4542,,C,0

978,3,"Barry, Miss. Julia",female,27.0,0,0,330844,7.8792,,Q,0

979,3,"Badman, Miss. Emily Louisa",female,18.0,0,0,A/4 31416,8.05,,S,1

980,3,"O'Donoghue, Ms. Bridget",female,,0,0,364856,7.75,,Q,0

981,2,"Wells, Master. Ralph Lester",male,2.0,1,1,29103,23.0,,S,1

982,3,"Dyker, Mrs. Adolf Fredrik (Anna Elisabeth Judith Andersson)",female,22.0,1,0,347072,13.9,,S,1

983,3,"Pedersen, Mr. Olaf",male,,0,0,345498,7.775,,S,0

984,1,"Davidson, Mrs. Thornton (Orian Hays)",female,27.0,1,2,F.C. 12750,52.0,B71,S,1

985,3,"Guest, Mr. Robert",male,,0,0,376563,8.05,,S,0

986,1,"Birnbaum, Mr. Jakob",male,25.0,0,0,13905,26.0,,C,0

987,3,"Tenglin, Mr. Gunnar Isidor",male,25.0,0,0,350033,7.7958,,S,1

988,1,"Cavendish, Mrs. Tyrell William (Julia Florence Siegel)",female,76.0,1,0,19877,78.85,C46,S,1

989,3,"Makinen, Mr. Kalle Edvard",male,29.0,0,0,STON/O 2. 3101268,7.925,,S,0

990,3,"Braf, Miss. Elin Ester Maria",female,20.0,0,0,347471,7.8542,,S,0

991,3,"Nancarrow, Mr. William Henry",male,33.0,0,0,A./5. 3338,8.05,,S,0

992,1,"Stengel, Mrs. Charles Emil Henry (Annie May Morris)",female,43.0,1,0,11778,55.4417,C116,C,1

993,2,"Weisz, Mr. Leopold",male,27.0,1,0,228414,26.0,,S,0

994,3,"Foley, Mr. William",male,,0,0,365235,7.75,,Q,0

995,3,"Johansson Palmquist, Mr. Oskar Leander",male,26.0,0,0,347070,7.775,,S,1

996,3,"Thomas, Mrs. Alexander (Thamine ""Thelma"")",female,16.0,1,1,2625,8.5167,,C,1

997,3,"Holthen, Mr. Johan Martin",male,28.0,0,0,C 4001,22.525,,S,0

998,3,"Buckley, Mr. Daniel",male,21.0,0,0,330920,7.8208,,Q,1

999,3,"Ryan, Mr. Edward",male,,0,0,383162,7.75,,Q,1

1000,3,"Willer, Mr. Aaron (""Abi Weller"")",male,,0,0,3410,8.7125,,S,0

1001,2,"Swane, Mr. George",male,18.5,0,0,248734,13.0,F,S,0

1002,2,"Stanton, Mr. Samuel Ward",male,41.0,0,0,237734,15.0458,,C,0

1003,3,"Shine, Miss. Ellen Natalia",female,,0,0,330968,7.7792,,Q,1

1004,1,"Evans, Miss. Edith Corse",female,36.0,0,0,PC 17531,31.6792,A29,C,0

1005,3,"Buckley, Miss. Katherine",female,18.5,0,0,329944,7.2833,,Q,0

1006,1,"Straus, Mrs. Isidor (Rosalie Ida Blun)",female,63.0,1,0,PC 17483,221.7792,C55 C57,S,0

1007,3,"Chronopoulos, Mr. Demetrios",male,18.0,1,0,2680,14.4542,,C,0

1008,3,"Thomas, Mr. John",male,,0,0,2681,6.4375,,C,0

1009,3,"Sandstrom, Miss. Beatrice Irene",female,1.0,1,1,PP 9549,16.7,G6,S,1

1010,1,"Beattie, Mr. Thomson",male,36.0,0,0,13050,75.2417,C6,C,0

1011,2,"Chapman, Mrs. John Henry (Sara Elizabeth Lawry)",female,29.0,1,0,SC/AH 29037,26.0,,S,0

1012,2,"Watt, Miss. Bertha J",female,12.0,0,0,C.A. 33595,15.75,,S,1

1013,3,"Kiernan, Mr. John",male,,1,0,367227,7.75,,Q,0

1014,1,"Schabert, Mrs. Paul (Emma Mock)",female,35.0,1,0,13236,57.75,C28,C,1

1015,3,"Carver, Mr. Alfred John",male,28.0,0,0,392095,7.25,,S,0

1016,3,"Kennedy, Mr. John",male,,0,0,368783,7.75,,Q,1

1017,3,"Cribb, Miss. Laura Alice",female,17.0,0,1,371362,16.1,,S,1

1018,3,"Brobeck, Mr. Karl Rudolf",male,22.0,0,0,350045,7.7958,,S,0

1019,3,"McCoy, Miss. Alicia",female,,2,0,367226,23.25,,Q,1

1020,2,"Bowenur, Mr. Solomon",male,42.0,0,0,211535,13.0,,S,0

1021,3,"Petersen, Mr. Marius",male,24.0,0,0,342441,8.05,,S,0

1022,3,"Spinner, Mr. Henry John",male,32.0,0,0,STON/OQ. 369943,8.05,,S,0

1023,1,"Gracie, Col. Archibald IV",male,53.0,0,0,113780,28.5,C51,C,1

1024,3,"Lefebre, Mrs. Frank (Frances)",female,,0,4,4133,25.4667,,S,0

1025,3,"Thomas, Mr. Charles P",male,,1,0,2621,6.4375,,C,0

1026,3,"Dintcheff, Mr. Valtcho",male,43.0,0,0,349226,7.8958,,S,0

1027,3,"Carlsson, Mr. Carl Robert",male,24.0,0,0,350409,7.8542,,S,0

1028,3,"Zakarian, Mr. Mapriededer",male,26.5,0,0,2656,7.225,,C,0

1029,2,"Schmidt, Mr. August",male,26.0,0,0,248659,13.0,,S,0

1030,3,"Drapkin, Miss. Jennie",female,23.0,0,0,SOTON/OQ 392083,8.05,,S,1

1031,3,"Goodwin, Mr. Charles Frederick",male,40.0,1,6,CA 2144,46.9,,S,0

1032,3,"Goodwin, Miss. Jessie Allis",female,10.0,5,2,CA 2144,46.9,,S,0

1033,1,"Daniels, Miss. Sarah",female,33.0,0,0,113781,151.55,,S,1

1034,1,"Ryerson, Mr. Arthur Larned",male,61.0,1,3,PC 17608,262.375,B57 B59 B63 B66,C,0

1035,2,"Beauchamp, Mr. Henry James",male,28.0,0,0,244358,26.0,,S,0

1036,1,"Lindeberg-Lind, Mr. Erik Gustaf (""Mr Edward Lingrey"")",male,42.0,0,0,17475,26.55,,S,0

1037,3,"Vander Planke, Mr. Julius",male,31.0,3,0,345763,18.0,,S,0

1038,1,"Hilliard, Mr. Herbert Henry",male,,0,0,17463,51.8625,E46,S,0

1039,3,"Davies, Mr. Evan",male,22.0,0,0,SC/A4 23568,8.05,,S,0

1040,1,"Crafton, Mr. John Bertram",male,,0,0,113791,26.55,,S,0

1041,2,"Lahtinen, Rev. William",male,30.0,1,1,250651,26.0,,S,0

1042,1,"Earnshaw, Mrs. Boulton (Olive Potter)",female,23.0,0,1,11767,83.1583,C54,C,1

1043,3,"Matinoff, Mr. Nicola",male,,0,0,349255,7.8958,,C,0

1044,3,"Storey, Mr. Thomas",male,60.5,0,0,3701,,,S,0

1045,3,"Klasen, Mrs. (Hulda Kristina Eugenia Lofqvist)",female,36.0,0,2,350405,12.1833,,S,0

1046,3,"Asplund, Master. Filip Oscar",male,13.0,4,2,347077,31.3875,,S,0

1047,3,"Duquemin, Mr. Joseph",male,24.0,0,0,S.O./P.P. 752,7.55,,S,1

1048,1,"Bird, Miss. Ellen",female,29.0,0,0,PC 17483,221.7792,C97,S,1

1049,3,"Lundin, Miss. Olga Elida",female,23.0,0,0,347469,7.8542,,S,1

1050,1,"Borebank, Mr. John James",male,42.0,0,0,110489,26.55,D22,S,0

1051,3,"Peacock, Mrs. Benjamin (Edith Nile)",female,26.0,0,2,SOTON/O.Q. 3101315,13.775,,S,0

1052,3,"Smyth, Miss. Julia",female,,0,0,335432,7.7333,,Q,1

1053,3,"Touma, Master. Georges Youssef",male,7.0,1,1,2650,15.2458,,C,1

1054,2,"Wright, Miss. Marion",female,26.0,0,0,220844,13.5,,S,1

1055,3,"Pearce, Mr. Ernest",male,,0,0,343271,7.0,,S,0

1056,2,"Peruschitz, Rev. Joseph Maria",male,41.0,0,0,237393,13.0,,S,0

1057,3,"Kink-Heilmann, Mrs. Anton (Luise Heilmann)",female,26.0,1,1,315153,22.025,,S,1

1058,1,"Brandeis, Mr. Emil",male,48.0,0,0,PC 17591,50.4958,B10,C,0

1059,3,"Ford, Mr. Edward Watson",male,18.0,2,2,W./C. 6608,34.375,,S,0

1060,1,"Cassebeer, Mrs. Henry Arthur Jr (Eleanor Genevieve Fosdick)",female,,0,0,17770,27.7208,,C,1

1061,3,"Hellstrom, Miss. Hilda Maria",female,22.0,0,0,7548,8.9625,,S,1

1062,3,"Lithman, Mr. Simon",male,,0,0,S.O./P.P. 251,7.55,,S,0

1063,3,"Zakarian, Mr. Ortin",male,27.0,0,0,2670,7.225,,C,0

1064,3,"Dyker, Mr. Adolf Fredrik",male,23.0,1,0,347072,13.9,,S,0

1065,3,"Torfa, Mr. Assad",male,,0,0,2673,7.2292,,C,0

1066,3,"Asplund, Mr. Carl Oscar Vilhelm Gustafsson",male,40.0,1,5,347077,31.3875,,S,0

1067,2,"Brown, Miss. Edith Eileen",female,15.0,0,2,29750,39.0,,S,1

1068,2,"Sincock, Miss. Maude",female,20.0,0,0,C.A. 33112,36.75,,S,1

1069,1,"Stengel, Mr. Charles Emil Henry",male,54.0,1,0,11778,55.4417,C116,C,1

1070,2,"Becker, Mrs. Allen Oliver (Nellie E Baumgardner)",female,36.0,0,3,230136,39.0,F4,S,1

1071,1,"Compton, Mrs. Alexander Taylor (Mary Eliza Ingersoll)",female,64.0,0,2,PC 17756,83.1583,E45,C,1

1072,2,"McCrie, Mr. James Matthew",male,30.0,0,0,233478,13.0,,S,0

1073,1,"Compton, Mr. Alexander Taylor Jr",male,37.0,1,1,PC 17756,83.1583,E52,C,0

1074,1,"Marvin, Mrs. Daniel Warner (Mary Graham Carmichael Farquarson)",female,18.0,1,0,113773,53.1,D30,S,1

1075,3,"Lane, Mr. Patrick",male,,0,0,7935,7.75,,Q,0

1076,1,"Douglas, Mrs. Frederick Charles (Mary Helene Baxter)",female,27.0,1,1,PC 17558,247.5208,B58 B60,C,1

1077,2,"Maybery, Mr. Frank Hubert",male,40.0,0,0,239059,16.0,,S,0

1078,2,"Phillips, Miss. Alice Frances Louisa",female,21.0,0,1,S.O./P.P. 2,21.0,,S,1

1079,3,"Davies, Mr. Joseph",male,17.0,2,0,A/4 48873,8.05,,S,0

1080,3,"Sage, Miss. Ada",female,,8,2,CA. 2343,69.55,,S,0

1081,2,"Veal, Mr. James",male,40.0,0,0,28221,13.0,,S,0

1082,2,"Angle, Mr. William A",male,34.0,1,0,226875,26.0,,S,0

1083,1,"Salomon, Mr. Abraham L",male,,0,0,111163,26.0,,S,1

1084,3,"van Billiard, Master. Walter John",male,11.5,1,1,A/5. 851,14.5,,S,0

1085,2,"Lingane, Mr. John",male,61.0,0,0,235509,12.35,,Q,0

1086,2,"Drew, Master. Marshall Brines",male,8.0,0,2,28220,32.5,,S,1

1087,3,"Karlsson, Mr. Julius Konrad Eugen",male,33.0,0,0,347465,7.8542,,S,0

1088,1,"Spedden, Master. Robert Douglas",male,6.0,0,2,16966,134.5,E34,C,1

1089,3,"Nilsson, Miss. Berta Olivia",female,18.0,0,0,347066,7.775,,S,1

1090,2,"Baimbrigge, Mr. Charles Robert",male,23.0,0,0,C.A. 31030,10.5,,S,0

1091,3,"Rasmussen, Mrs. (Lena Jacobsen Solvang)",female,,0,0,65305,8.1125,,S,0

1092,3,"Murphy, Miss. Nora",female,,0,0,36568,15.5,,Q,1

1093,3,"Danbom, Master. Gilbert Sigvard Emanuel",male,0.33,0,2,347080,14.4,,S,0

1094,1,"Astor, Col. John Jacob",male,47.0,1,0,PC 17757,227.525,C62 C64,C,0

1095,2,"Quick, Miss. Winifred Vera",female,8.0,1,1,26360,26.0,,S,1

1096,2,"Andrew, Mr. Frank Thomas",male,25.0,0,0,C.A. 34050,10.5,,S,0

1097,1,"Omont, Mr. Alfred Fernand",male,,0,0,F.C. 12998,25.7417,,C,1

1098,3,"McGowan, Miss. Katherine",female,35.0,0,0,9232,7.75,,Q,0

1099,2,"Collett, Mr. Sidney C Stuart",male,24.0,0,0,28034,10.5,,S,1

1100,1,"Rosenbaum, Miss. Edith Louise",female,33.0,0,0,PC 17613,27.7208,A11,C,1

1101,3,"Delalic, Mr. Redjo",male,25.0,0,0,349250,7.8958,,S,0

1102,3,"Andersen, Mr. Albert Karvin",male,32.0,0,0,C 4001,22.525,,S,0

1103,3,"Finoli, Mr. Luigi",male,,0,0,SOTON/O.Q. 3101308,7.05,,S,1

1104,2,"Deacon, Mr. Percy William",male,17.0,0,0,S.O.C. 14879,73.5,,S,0

1105,2,"Howard, Mrs. Benjamin (Ellen Truelove Arman)",female,60.0,1,0,24065,26.0,,S,0

1106,3,"Andersson, Miss. Ida Augusta Margareta",female,38.0,4,2,347091,7.775,,S,0

1107,1,"Head, Mr. Christopher",male,42.0,0,0,113038,42.5,B11,S,0

1108,3,"Mahon, Miss. Bridget Delia",female,,0,0,330924,7.8792,,Q,0

1109,1,"Wick, Mr. George Dennick",male,57.0,1,1,36928,164.8667,,S,0

1110,1,"Widener, Mrs. George Dunton (Eleanor Elkins)",female,50.0,1,1,113503,211.5,C80,C,1

1111,3,"Thomson, Mr. Alexander Morrison",male,,0,0,32302,8.05,,S,0

1112,2,"Duran y More, Miss. Florentina",female,30.0,1,0,SC/PARIS 2148,13.8583,,C,1

1113,3,"Reynolds, Mr. Harold J",male,21.0,0,0,342684,8.05,,S,0

1114,2,"Cook, Mrs. (Selena Rogers)",female,22.0,0,0,W./C. 14266,10.5,F33,S,1

1115,3,"Karlsson, Mr. Einar Gervasius",male,21.0,0,0,350053,7.7958,,S,1

1116,1,"Candee, Mrs. Edward (Helen Churchill Hungerford)",female,53.0,0,0,PC 17606,27.4458,,C,1

1117,3,"Moubarek, Mrs. George (Omine ""Amenia"" Alexander)",female,,0,2,2661,15.2458,,C,1

1118,3,"Asplund, Mr. Johan Charles",male,23.0,0,0,350054,7.7958,,S,1

1119,3,"McNeill, Miss. Bridget",female,,0,0,370368,7.75,,Q,0

1120,3,"Everett, Mr. Thomas James",male,40.5,0,0,C.A. 6212,15.1,,S,0

1121,2,"Hocking, Mr. Samuel James Metcalfe",male,36.0,0,0,242963,13.0,,S,0

1122,2,"Sweet, Mr. George Frederick",male,14.0,0,0,220845,65.0,,S,0

1123,1,"Willard, Miss. Constance",female,21.0,0,0,113795,26.55,,S,1

1124,3,"Wiklund, Mr. Karl Johan",male,21.0,1,0,3101266,6.4958,,S,0

1125,3,"Linehan, Mr. Michael",male,,0,0,330971,7.8792,,Q,0

1126,1,"Cumings, Mr. John Bradley",male,39.0,1,0,PC 17599,71.2833,C85,C,0

1127,3,"Vendel, Mr. Olof Edvin",male,20.0,0,0,350416,7.8542,,S,0

1128,1,"Warren, Mr. Frank Manley",male,64.0,1,0,110813,75.25,D37,C,0

1129,3,"Baccos, Mr. Raffull",male,20.0,0,0,2679,7.225,,C,0

1130,2,"Hiltunen, Miss. Marta",female,18.0,1,1,250650,13.0,,S,0

1131,1,"Douglas, Mrs. Walter Donald (Mahala Dutton)",female,48.0,1,0,PC 17761,106.425,C86,C,1

1132,1,"Lindstrom, Mrs. Carl Johan (Sigrid Posse)",female,55.0,0,0,112377,27.7208,,C,1

1133,2,"Christy, Mrs. (Alice Frances)",female,45.0,0,2,237789,30.0,,S,1

1134,1,"Spedden, Mr. Frederic Oakley",male,45.0,1,1,16966,134.5,E34,C,1

1135,3,"Hyman, Mr. Abraham",male,,0,0,3470,7.8875,,S,1

1136,3,"Johnston, Master. William Arthur ""Willie""",male,,1,2,W./C. 6607,23.45,,S,0

1137,1,"Kenyon, Mr. Frederick R",male,41.0,1,0,17464,51.8625,D21,S,0

1138,2,"Karnes, Mrs. J Frank (Claire Bennett)",female,22.0,0,0,F.C.C. 13534,21.0,,S,0

1139,2,"Drew, Mr. James Vivian",male,42.0,1,1,28220,32.5,,S,0

1140,2,"Hold, Mrs. Stephen (Annie Margaret Hill)",female,29.0,1,0,26707,26.0,,S,1

1141,3,"Khalil, Mrs. Betros (Zahie ""Maria"" Elias)",female,,1,0,2660,14.4542,,C,0

1142,2,"West, Miss. Barbara J",female,0.92,1,2,C.A. 34651,27.75,,S,1

1143,3,"Abrahamsson, Mr. Abraham August Johannes",male,20.0,0,0,SOTON/O2 3101284,7.925,,S,1

1144,1,"Clark, Mr. Walter Miller",male,27.0,1,0,13508,136.7792,C89,C,0

1145,3,"Salander, Mr. Karl Johan",male,24.0,0,0,7266,9.325,,S,0

1146,3,"Wenzel, Mr. Linhart",male,32.5,0,0,345775,9.5,,S,0

1147,3,"MacKay, Mr. George William",male,,0,0,C.A. 42795,7.55,,S,0

1148,3,"Mahon, Mr. John",male,,0,0,AQ/4 3130,7.75,,Q,0

1149,3,"Niklasson, Mr. Samuel",male,28.0,0,0,363611,8.05,,S,0

1150,2,"Bentham, Miss. Lilian W",female,19.0,0,0,28404,13.0,,S,1

1151,3,"Midtsjo, Mr. Karl Albert",male,21.0,0,0,345501,7.775,,S,1

1152,3,"de Messemaeker, Mr. Guillaume Joseph",male,36.5,1,0,345572,17.4,,S,1

1153,3,"Nilsson, Mr. August Ferdinand",male,21.0,0,0,350410,7.8542,,S,0

1154,2,"Wells, Mrs. Arthur Henry (""Addie"" Dart Trevaskis)",female,29.0,0,2,29103,23.0,,S,1

1155,3,"Klasen, Miss. Gertrud Emilia",female,1.0,1,1,350405,12.1833,,S,0

1156,2,"Portaluppi, Mr. Emilio Ilario Giuseppe",male,30.0,0,0,C.A. 34644,12.7375,,C,1

1157,3,"Lyntakoff, Mr. Stanko",male,,0,0,349235,7.8958,,S,0

1158,1,"Chisholm, Mr. Roderick Robert Crispin",male,,0,0,112051,0.0,,S,0

1159,3,"Warren, Mr. Charles William",male,,0,0,C.A. 49867,7.55,,S,0

1160,3,"Howard, Miss. May Elizabeth",female,,0,0,A. 2. 39186,8.05,,S,1

1161,3,"Pokrnic, Mr. Mate",male,17.0,0,0,315095,8.6625,,S,0

1162,1,"McCaffry, Mr. Thomas Francis",male,46.0,0,0,13050,75.2417,C6,C,0

1163,3,"Fox, Mr. Patrick",male,,0,0,368573,7.75,,Q,0

1164,1,"Clark, Mrs. Walter Miller (Virginia McDowell)",female,26.0,1,0,13508,136.7792,C89,C,1

1165,3,"Lennon, Miss. Mary",female,,1,0,370371,15.5,,Q,0

1166,3,"Saade, Mr. Jean Nassr",male,,0,0,2676,7.225,,C,0

1167,2,"Bryhl, Miss. Dagmar Jenny Ingeborg ",female,20.0,1,0,236853,26.0,,S,1

1168,2,"Parker, Mr. Clifford Richard",male,28.0,0,0,SC 14888,10.5,,S,0

1169,2,"Faunthorpe, Mr. Harry",male,40.0,1,0,2926,26.0,,S,0

1170,2,"Ware, Mr. John James",male,30.0,1,0,CA 31352,21.0,,S,0

1171,2,"Oxenham, Mr. Percy Thomas",male,22.0,0,0,W./C. 14260,10.5,,S,1

1172,3,"Oreskovic, Miss. Jelka",female,23.0,0,0,315085,8.6625,,S,0

1173,3,"Peacock, Master. Alfred Edward",male,0.75,1,1,SOTON/O.Q. 3101315,13.775,,S,0

1174,3,"Fleming, Miss. Honora",female,,0,0,364859,7.75,,Q,0

1175,3,"Touma, Miss. Maria Youssef",female,9.0,1,1,2650,15.2458,,C,1

1176,3,"Rosblom, Miss. Salli Helena",female,2.0,1,1,370129,20.2125,,S,0

1177,3,"Dennis, Mr. William",male,36.0,0,0,A/5 21175,7.25,,S,0

1178,3,"Franklin, Mr. Charles (Charles Fardon)",male,,0,0,SOTON/O.Q. 3101314,7.25,,S,0

1179,1,"Snyder, Mr. John Pillsbury",male,24.0,1,0,21228,82.2667,B45,S,1

1180,3,"Mardirosian, Mr. Sarkis",male,,0,0,2655,7.2292,F E46,C,0

1181,3,"Ford, Mr. Arthur",male,,0,0,A/5 1478,8.05,,S,0

1182,1,"Rheims, Mr. George Alexander Lucien",male,,0,0,PC 17607,39.6,,S,1

1183,3,"Daly, Miss. Margaret Marcella ""Maggie""",female,30.0,0,0,382650,6.95,,Q,1

1184,3,"Nasr, Mr. Mustafa",male,,0,0,2652,7.2292,,C,0

1185,1,"Dodge, Dr. Washington",male,53.0,1,1,33638,81.8583,A34,S,1

1186,3,"Wittevrongel, Mr. Camille",male,36.0,0,0,345771,9.5,,S,0

1187,3,"Angheloff, Mr. Minko",male,26.0,0,0,349202,7.8958,,S,0

1188,2,"Laroche, Miss. Louise",female,1.0,1,2,SC/Paris 2123,41.5792,,C,1

1189,3,"Samaan, Mr. Hanna",male,,2,0,2662,21.6792,,C,0

1190,1,"Loring, Mr. Joseph Holland",male,30.0,0,0,113801,45.5,,S,0

1191,3,"Johansson, Mr. Nils",male,29.0,0,0,347467,7.8542,,S,0

1192,3,"Olsson, Mr. Oscar Wilhelm",male,32.0,0,0,347079,7.775,,S,1

1193,2,"Malachard, Mr. Noel",male,,0,0,237735,15.0458,D,C,0

1194,2,"Phillips, Mr. Escott Robert",male,43.0,0,1,S.O./P.P. 2,21.0,,S,0

1195,3,"Pokrnic, Mr. Tome",male,24.0,0,0,315092,8.6625,,S,0

1196,3,"McCarthy, Miss. Catherine ""Katie""",female,,0,0,383123,7.75,,Q,1

1197,1,"Crosby, Mrs. Edward Gifford (Catherine Elizabeth Halstead)",female,64.0,1,1,112901,26.55,B26,S,1

1198,1,"Allison, Mr. Hudson Joshua Creighton",male,30.0,1,2,113781,151.55,C22 C26,S,0

1199,3,"Aks, Master. Philip Frank",male,0.83,0,1,392091,9.35,,S,1

1200,1,"Hays, Mr. Charles Melville",male,55.0,1,1,12749,93.5,B69,S,0

1201,3,"Hansen, Mrs. Claus Peter (Jennie L Howard)",female,45.0,1,0,350026,14.1083,,S,1

1202,3,"Cacic, Mr. Jego Grga",male,18.0,0,0,315091,8.6625,,S,0

1203,3,"Vartanian, Mr. David",male,22.0,0,0,2658,7.225,,C,1

1204,3,"Sadowitz, Mr. Harry",male,,0,0,LP 1588,7.575,,S,0

1205,3,"Carr, Miss. Jeannie",female,37.0,0,0,368364,7.75,,Q,0

1206,1,"White, Mrs. John Stuart (Ella Holmes)",female,55.0,0,0,PC 17760,135.6333,C32,C,1

1207,3,"Hagardon, Miss. Kate",female,17.0,0,0,AQ/3. 30631,7.7333,,Q,0

1208,1,"Spencer, Mr. William Augustus",male,57.0,1,0,PC 17569,146.5208,B78,C,0

1209,2,"Rogers, Mr. Reginald Harry",male,19.0,0,0,28004,10.5,,S,0

1210,3,"Jonsson, Mr. Nils Hilding",male,27.0,0,0,350408,7.8542,,S,0

1211,2,"Jefferys, Mr. Ernest Wilfred",male,22.0,2,0,C.A. 31029,31.5,,S,0

1212,3,"Andersson, Mr. Johan Samuel",male,26.0,0,0,347075,7.775,,S,0

1213,3,"Krekorian, Mr. Neshan",male,25.0,0,0,2654,7.2292,F E57,C,1

1214,2,"Nesson, Mr. Israel",male,26.0,0,0,244368,13.0,F2,S,0

1215,1,"Rowe, Mr. Alfred G",male,33.0,0,0,113790,26.55,,S,0

1216,1,"Kreuchen, Miss. Emilie",female,39.0,0,0,24160,211.3375,,S,1

1217,3,"Assam, Mr. Ali",male,23.0,0,0,SOTON/O.Q. 3101309,7.05,,S,0

1218,2,"Becker, Miss. Ruth Elizabeth",female,12.0,2,1,230136,39.0,F4,S,1

1219,1,"Rosenshine, Mr. George (""Mr George Thorne"")",male,46.0,0,0,PC 17585,79.2,,C,0

1220,2,"Clarke, Mr. Charles Valentine",male,29.0,1,0,2003,26.0,,S,0

1221,2,"Enander, Mr. Ingvar",male,21.0,0,0,236854,13.0,,S,0

1222,2,"Davies, Mrs. John Morgan (Elizabeth Agnes Mary White) ",female,48.0,0,2,C.A. 33112,36.75,,S,1

1223,1,"Dulles, Mr. William Crothers",male,39.0,0,0,PC 17580,29.7,A18,C,0

1224,3,"Thomas, Mr. Tannous",male,,0,0,2684,7.225,,C,0

1225,3,"Nakid, Mrs. Said (Waika ""Mary"" Mowad)",female,19.0,1,1,2653,15.7417,,C,1

1226,3,"Cor, Mr. Ivan",male,27.0,0,0,349229,7.8958,,S,0

1227,1,"Maguire, Mr. John Edward",male,30.0,0,0,110469,26.0,C106,S,0

1228,2,"de Brito, Mr. Jose Joaquim",male,32.0,0,0,244360,13.0,,S,0

1229,3,"Elias, Mr. Joseph",male,39.0,0,2,2675,7.2292,,C,0

1230,2,"Denbury, Mr. Herbert",male,25.0,0,0,C.A. 31029,31.5,,S,0

1231,3,"Betros, Master. Seman",male,,0,0,2622,7.2292,,C,0

1232,2,"Fillbrook, Mr. Joseph Charles",male,18.0,0,0,C.A. 15185,10.5,,S,0

1233,3,"Lundstrom, Mr. Thure Edvin",male,32.0,0,0,350403,7.5792,,S,1

1234,3,"Sage, Mr. John George",male,,1,9,CA. 2343,69.55,,S,0

1235,1,"Cardeza, Mrs. James Warburton Martinez (Charlotte Wardle Drake)",female,58.0,0,1,PC 17755,512.3292,B51 B53 B55,C,1

1236,3,"van Billiard, Master. James William",male,,1,1,A/5. 851,14.5,,S,0

1237,3,"Abelseth, Miss. Karen Marie",female,16.0,0,0,348125,7.65,,S,1

1238,2,"Botsford, Mr. William Hull",male,26.0,0,0,237670,13.0,,S,0

1239,3,"Whabee, Mrs. George Joseph (Shawneene Abi-Saab)",female,38.0,0,0,2688,7.2292,,C,1

1240,2,"Giles, Mr. Ralph",male,24.0,0,0,248726,13.5,,S,0

1241,2,"Walcroft, Miss. Nellie",female,31.0,0,0,F.C.C. 13528,21.0,,S,1

1242,1,"Greenfield, Mrs. Leo David (Blanche Strouse)",female,45.0,0,1,PC 17759,63.3583,D10 D12,C,1

1243,2,"Stokes, Mr. Philip Joseph",male,25.0,0,0,F.C.C. 13540,10.5,,S,0

1244,2,"Dibden, Mr. William",male,18.0,0,0,S.O.C. 14879,73.5,,S,0

1245,2,"Herman, Mr. Samuel",male,49.0,1,2,220845,65.0,,S,0

1246,3,"Dean, Miss. Elizabeth Gladys ""Millvina""",female,0.17,1,2,C.A. 2315,20.575,,S,1

1247,1,"Julian, Mr. Henry Forbes",male,50.0,0,0,113044,26.0,E60,S,0

1248,1,"Brown, Mrs. John Murray (Caroline Lane Lamson)",female,59.0,2,0,11769,51.4792,C101,S,1

1249,3,"Lockyer, Mr. Edward",male,,0,0,1222,7.8792,,S,0

1250,3,"O'Keefe, Mr. Patrick",male,,0,0,368402,7.75,,Q,1

1251,3,"Lindell, Mrs. Edvard Bengtsson (Elin Gerda Persson)",female,30.0,1,0,349910,15.55,,S,0

1252,3,"Sage, Master. William Henry",male,14.5,8,2,CA. 2343,69.55,,S,0

1253,2,"Mallet, Mrs. Albert (Antoinette Magnin)",female,24.0,1,1,S.C./PARIS 2079,37.0042,,C,1

1254,2,"Ware, Mrs. John James (Florence Louise Long)",female,31.0,0,0,CA 31352,21.0,,S,1

1255,3,"Strilic, Mr. Ivan",male,27.0,0,0,315083,8.6625,,S,0

1256,1,"Harder, Mrs. George Achilles (Dorothy Annan)",female,25.0,1,0,11765,55.4417,E50,C,1

1257,3,"Sage, Mrs. John (Annie Bullen)",female,,1,9,CA. 2343,69.55,,S,0

1258,3,"Caram, Mr. Joseph",male,,1,0,2689,14.4583,,C,0

1259,3,"Riihivouri, Miss. Susanna Juhantytar ""Sanni""",female,22.0,0,0,3101295,39.6875,,S,0

1260,1,"Gibson, Mrs. Leonard (Pauline C Boeson)",female,45.0,0,1,112378,59.4,,C,1

1261,2,"Pallas y Castello, Mr. Emilio",male,29.0,0,0,SC/PARIS 2147,13.8583,,C,1

1262,2,"Giles, Mr. Edgar",male,21.0,1,0,28133,11.5,,S,0

1263,1,"Wilson, Miss. Helen Alice",female,31.0,0,0,16966,134.5,E39 E41,C,1

1264,1,"Ismay, Mr. Joseph Bruce",male,49.0,0,0,112058,0.0,B52 B54 B56,S,1

1265,2,"Harbeck, Mr. William H",male,44.0,0,0,248746,13.0,,S,0

1266,1,"Dodge, Mrs. Washington (Ruth Vidaver)",female,54.0,1,1,33638,81.8583,A34,S,1

1267,1,"Bowen, Miss. Grace Scott",female,45.0,0,0,PC 17608,262.375,,C,1

1268,3,"Kink, Miss. Maria",female,22.0,2,0,315152,8.6625,,S,0

1269,2,"Cotterill, Mr. Henry ""Harry""",male,21.0,0,0,29107,11.5,,S,0

1270,1,"Hipkins, Mr. William Edward",male,55.0,0,0,680,50.0,C39,S,0

1271,3,"Asplund, Master. Carl Edgar",male,5.0,4,2,347077,31.3875,,S,0

1272,3,"O'Connor, Mr. Patrick",male,,0,0,366713,7.75,,Q,0

1273,3,"Foley, Mr. Joseph",male,26.0,0,0,330910,7.8792,,Q,0

1274,3,"Risien, Mrs. Samuel (Emma)",female,,0,0,364498,14.5,,S,0

1275,3,"McNamee, Mrs. Neal (Eileen O'Leary)",female,19.0,1,0,376566,16.1,,S,0

1276,2,"Wheeler, Mr. Edwin ""Frederick""",male,,0,0,SC/PARIS 2159,12.875,,S,0

1277,2,"Herman, Miss. Kate",female,24.0,1,2,220845,65.0,,S,1

1278,3,"Aronsson, Mr. Ernst Axel Algot",male,24.0,0,0,349911,7.775,,S,0

1279,2,"Ashby, Mr. John",male,57.0,0,0,244346,13.0,,S,0

1280,3,"Canavan, Mr. Patrick",male,21.0,0,0,364858,7.75,,Q,0

1281,3,"Palsson, Master. Paul Folke",male,6.0,3,1,349909,21.075,,S,0

1282,1,"Payne, Mr. Vivian Ponsonby",male,23.0,0,0,12749,93.5,B24,S,0

1283,1,"Lines, Mrs. Ernest H (Elizabeth Lindsey James)",female,51.0,0,1,PC 17592,39.4,D28,S,1

1284,3,"Abbott, Master. Eugene Joseph",male,13.0,0,2,C.A. 2673,20.25,,S,0

1285,2,"Gilbert, Mr. William",male,47.0,0,0,C.A. 30769,10.5,,S,0

1286,3,"Kink-Heilmann, Mr. Anton",male,29.0,3,1,315153,22.025,,S,1

1287,1,"Smith, Mrs. Lucien Philip (Mary Eloise Hughes)",female,18.0,1,0,13695,60.0,C31,S,1

1288,3,"Colbert, Mr. Patrick",male,24.0,0,0,371109,7.25,,Q,0

1289,1,"Frolicher-Stehli, Mrs. Maxmillian (Margaretha Emerentia Stehli)",female,48.0,1,1,13567,79.2,B41,C,1

1290,3,"Larsson-Rondberg, Mr. Edvard A",male,22.0,0,0,347065,7.775,,S,0

1291,3,"Conlon, Mr. Thomas Henry",male,31.0,0,0,21332,7.7333,,Q,0

1292,1,"Bonnell, Miss. Caroline",female,30.0,0,0,36928,164.8667,C7,S,1

1293,2,"Gale, Mr. Harry",male,38.0,1,0,28664,21.0,,S,0

1294,1,"Gibson, Miss. Dorothy Winifred",female,22.0,0,1,112378,59.4,,C,1

1295,1,"Carrau, Mr. Jose Pedro",male,17.0,0,0,113059,47.1,,S,0

1296,1,"Frauenthal, Mr. Isaac Gerald",male,43.0,1,0,17765,27.7208,D40,C,1

1297,2,"Nourney, Mr. Alfred (""Baron von Drachstedt"")",male,20.0,0,0,SC/PARIS 2166,13.8625,D38,C,1

1298,2,"Ware, Mr. William Jeffery",male,23.0,1,0,28666,10.5,,S,0

1299,1,"Widener, Mr. George Dunton",male,50.0,1,1,113503,211.5,C80,C,0

1300,3,"Riordan, Miss. Johanna ""Hannah""",female,,0,0,334915,7.7208,,Q,1

1301,3,"Peacock, Miss. Treasteall",female,3.0,1,1,SOTON/O.Q. 3101315,13.775,,S,0

1302,3,"Naughton, Miss. Hannah",female,,0,0,365237,7.75,,Q,0

1303,1,"Minahan, Mrs. William Edward (Lillian E Thorpe)",female,37.0,1,0,19928,90.0,C78,Q,1

1304,3,"Henriksson, Miss. Jenny Lovisa",female,28.0,0,0,347086,7.775,,S,0

1305,3,"Spector, Mr. Woolf",male,,0,0,A.5. 3236,8.05,,S,0

1306,1,"Oliva y Ocana, Dona. Fermina",female,39.0,0,0,PC 17758,108.9,C105,C,1

1307,3,"Saether, Mr. Simon Sivertsen",male,38.5,0,0,SOTON/O.Q. 3101262,7.25,,S,0

1308,3,"Ware, Mr. Frederick",male,,0,0,359309,8.05,,S,0

1309,3,"Peter, Master. Michael J",male,,1,1,2668,22.3583,,C,1'''
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# We'll use this for the 'cheating' or 'reverse engineering' part where we'll compare the pub LB

# score of our submissions with the actual CV score on the complete dataset.

complete_df = pd.concat([pd.read_csv('../input/train.csv'),

                         pd.read_csv(StringIO(test_complete_csv_string))])

# Same order please

complete_df = complete_df[train_df.columns]

train_df.head()
# Let's just dive in and get a feeling of how to use hallucinate to quickly get a feel

# of the dataset

# The experiment definition below is pretty straight forward, we're defining a 'Kaggle'

# experiment with 30 fold stratified cross validation, no CV data shuffling, accuracy score

# and which will run estimator predictions in parallel. hallucinate parallelizes classifiers,

# every classifier gets its own process

# UPDATE: I've learned while running this kernel that Kaggle's resources for a

# kernel are pretty scarce so it may be useless to parallelize.

kaggle_all = Experiment(name='Kaggle', cv=30, cv_shuffle=False, sc='accuracy', parallel=False)



# Let's just use all the features as of now, just drop those not helpful as features

feature_names = train_df.columns.drop(['PassengerId', 'Survived']).values.tolist()



# Filter categoricals

categorical_features = train_df.select_dtypes(include=[object]).columns.values.tolist()



# Create the first feature set

kaggle_features = kaggle_all.make_features(feature_names, name='All Columns',

                                           train_data=train_df, test_data=test_df,

                                           target='Survived')



# Get a quick overview of the experiment

kaggle_all.overview()
kaggle_all.show_null_stats()
# This is hallucinate's feature engineering API, you apply all sorts of transforms to the

# existing (or non existing for that matter) features. Below we're only doing feature imputation

# by filling the non existent (fillna in Pandas lingo) values for those columns with the

# specified value

kaggle_features.transform(['Cabin'], 'fillna', strategy='value', value='XXX')



# Will fill in the missing Age values with the Age median

kaggle_features.transform(['Age'], 'fillna', strategy='median')



# Will fill in the missing Fare values with the average Fare

kaggle_features.transform(['Fare'], 'fillna', strategy='mean')



# Will fill in the missing Embarked values with the most frequent value, you can get back quickly

# to the overview to see that S is where most of the passengers embarked

kaggle_features.transform(['Embarked'], 'fillna', strategy='value', value='S')



# One-hot encode categorical features

kaggle_features.transform(categorical_features, 'onehot')

kaggle_features
kaggle_all.overview()
kaggle_all.show_null_stats(preprocess=True)
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



# Add some estimator configurations to the experiment

kaggle_all.add_estimator(EstimatorConfig(LGBMClassifier(), {}, 'LGB'))

kaggle_all.add_estimator(EstimatorConfig(XGBClassifier(), {}, 'XGB'))

kaggle_all.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LRE'))



# Run grid search on all of them. Grid search params to be specified in the now empty dict {}

kaggle_all.grid_search_all()
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Oops, there's an ugly warning right there, most probably caused by seaborn using a deprecated

# API, hopefully it'll go away as we upgrade the package versions and the awesome seaborn guys

# fix it. I'll suppress it for the sake of visualisation right there.

kaggle_all.plot_cv_runs()
# There's some fishy thing happening with the PassengerId being written as a String column

# for the submissions so just turn everything to int and overwrite

import os

def hack_to_fix_submissions(folder):

    for submission_name in os.listdir(folder):

        df = pd.read_csv('{}/{}'.format(folder, submission_name))

        df = df.astype(int)

        df.to_csv('{}/{}'.format(folder, submission_name), index=False)

    print(' -- Submissions in folder {} fixed!'.format(folder))



kaggle = Kaggle(experiment=kaggle_all, index_feature='PassengerId', target_feature='Survived')

submissions = kaggle.create_submissions(out_folder='kaggle_all')

hack_to_fix_submissions('kaggle_all')
from sklearn.metrics import accuracy_score



def evaluate_on_real_data(_model, _features):

    X_train, y_train, _, _ = _features.build_Xy()

    X_test, _, _ = _features.build_X(for_test=True)

    _model.fit(X_train, y_train)

    predictions = _model.predict(X_test).astype(int)

    real_values = pd.read_csv(StringIO(test_complete_csv_string))['Survived'].values

    return accuracy_score(real_values, predictions)



def compare_last_run_with_real_data(experiment):

    estimator_names = [c.name for c in experiment.configs]

    feature_set_names = [f.name for f in experiment.features_sources]

    for features in feature_set_names:

        for estimator in estimator_names:

            cv_run = experiment.find_runs(estimator, features)[-1]

            test_score = evaluate_on_real_data(cv_run.best_model, experiment.find_features(features))

            training_score = cv_run.mean_score

            print(' o {} / {}: Training: {:.4f}, Test: {:.4f}'.format(estimator, features, training_score, test_score))



compare_last_run_with_real_data(kaggle_all)
kaggle_all.find_best_model('XGB')
from sklearn.feature_selection import SelectFromModel

# Let's chose a neutral classifier as to not favor one of our classifiers over the other,

# just as an exercise, so we'll ask a DecisionTreeClassifier to make the selection

kaggle_features.set_feature_selector(SelectFromModel(DecisionTreeClassifier(random_state=7)))

kaggle_all.grid_search_all()

kaggle_all.plot_cv_runs()
compare_last_run_with_real_data(kaggle_all)
kaggle_1 = Experiment(name='Kaggle 1', cv=30, cv_shuffle=False, sc='accuracy', parallel=True)

features_1 = kaggle_1.make_features(feature_names, train_data=train_df, test_data=test_df,

                                   target='Survived')

features_1.transform(['Cabin'], 'fillna', strategy='value', value='XXX')

# It would be worth considering filling the missing Age values by prediction or something smarter

# Let's keep it simple for now

features_1.transform(['Age'], 'fillna', strategy='median')

features_1.transform(['Fare'], 'fillna', strategy='mean')

features_1.transform(['Embarked'], 'fillna', strategy='value', value='S')



# This call allows you to tap into the internals of hallucinate and retrieve

# the 'pre-processed' features just as they are after hallucinate applies all

# the transformations defined on them

train_df1 = features_1.preprocess(include_one_hot=False)

# Re-append Survived to aid in our (custom) visualization quest

train_df1 = pd.concat([train_df1, train_df[['Survived']]], axis=1)
sns.factorplot(x='Pclass', data=train_df1, hue='Survived', kind='count',

               palette=['darkred', 'green'])
import random

real_values = pd.read_csv(StringIO(test_complete_csv_string))['Survived'].values

predictions = test_df['Pclass'].apply(lambda x: random.randint(0, 1) if x == 2 else 1 if x == 1 else 0)

print('Pclass score: {:.4f}'.format(accuracy_score(real_values, predictions)))
sns.factorplot(x='Sex', data=train_df1, hue='Survived', kind='count',

               palette=['darkred', 'green'])
sns.factorplot(x='Pclass', data=train_df1, hue='Survived', col='Sex', kind='count',

               palette=['darkred', 'green'])
def predict(tpl):

    sex, pclass = tpl

    if sex == 'female':

        if pclass < 3: # women in class 1 and 2 survived

            return 1

        else:

            return random.randint(0, 1) # 50% survival chance for women in class 3

    else: # male

        return 0

predictions = test_df[['Sex', 'Pclass']].apply(predict, axis=1)

print('Pclass + Sex score: {:.4f}'.format(accuracy_score(real_values, predictions)))
sns.factorplot(x='Pclass', y='Age', data=train_df1, hue='Survived', col='Sex', kind='box',

               palette=['darkred', 'green'])
sns.factorplot(x='Pclass', y='Age', data=train_df1[train_df1['Age'] < 10], hue='Survived',

               col='Sex', kind='box',palette=['darkred', 'green'])
sns.factorplot(x='Pclass', y='Age', data=train_df1[train_df1['Age'] < 18], hue='Survived',

               col='Sex', kind='box',palette=['darkred', 'green'])
def predict(tpl):

    sex, pclass, age = tpl

    if sex == 'female':

        if pclass < 3: # women in class 1 and 2 survived

            return 1

        else:

            return random.randint(0, 1) # 50% survival chance for women in class 3

    else:

        if pclass < 3 and age < 18: # males under 18 in class 1 and 2

            return 1

        else:

            return 0

predictions = test_df[['Sex', 'Pclass', 'Age']].apply(predict, axis=1)

print('Pclass + Sex + Young males score: {:.4f}'.format(accuracy_score(real_values, predictions)))
train_df1[['Age']].describe()
import matplotlib.pyplot as plt

sns.distplot(train_df1[train_df1['Sex']=='male']['Age'], color='darkred', kde=False)

sns.distplot(train_df1[train_df1['Sex']=='female']['Age'], color='darkgreen', kde=False)

plt.gca().grid(False)
sns.distplot(train_df1[train_df1['Survived']==0]['Age'], color='darkred', kde=False)

sns.distplot(train_df1[train_df1['Survived']==1]['Age'], color='darkgreen', kde=False)

plt.gca().grid(False)
sns.factorplot(x='Sex', y='Fare', hue='Survived', kind='point',

               data=train_df1, palette=['darkred', 'green'])
sns.lmplot(x='Age', y='Fare', hue='Survived', col='Sex', fit_reg=False,

           data=train_df1[train_df1['Fare'] < 500],

           palette=['darkred', 'green'])
sns.distplot(train_df1['Fare'], bins=10, kde=False)

plt.gca().grid(False)
sns.factorplot(x='Pclass', hue='Survived', col='SibSp', kind='count', 

               data=train_df1[(train_df['Sex'] == 'male')],

               palette=['darkred', 'green'])
sns.factorplot(x='Pclass', hue='Survived', col='SibSp', kind='count', 

               data=train_df1[(train_df['Sex'] == 'female')],

               palette=['darkred', 'green'])
sns.factorplot(x='Parch', hue='Survived', col='SibSp', kind='count', 

               data=train_df1[(train_df['Sex'] == 'male')],

               palette=['darkred', 'green'])
sns.factorplot(x='Pclass', hue='Survived', col='SibSp', kind='count', 

               data=train_df1[(train_df['Sex'] == 'female')],

               palette=['darkred', 'green'])
sns.factorplot(x='Embarked', data=train_df1, hue='Survived', col='Sex', kind='count',

               palette=['darkred', 'green'])
sns.factorplot(x='Embarked', y='Fare', data=train_df1, hue='Survived', col='Sex',

               palette=['darkred', 'green'])
# Create the experiment

titanic = Experiment(name='Titanic', cv=30, cv_shuffle=False, sc='accuracy', parallel=False)

titanic.add_estimator(EstimatorConfig(LGBMClassifier(), {}, 'LGB'))

titanic.add_estimator(EstimatorConfig(XGBClassifier(), {}, 'XGB'))

titanic.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LRE'))



# Let's call them 'Bare' features, those that need a minimal amount of work such as fillna,

# and onehot or not at all

bare_feature_names = ['Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Age', 'Embarked']

def add_bare_features(experiment):

    bare_features = experiment.make_features(bare_feature_names, name='Bare features',

                                                train_data=train_df1,test_data=test_df,

                                                target='Survived')

    bare_features.transform(['Age'], 'fillna', strategy='median')

    bare_features.transform(['Fare'], 'fillna', strategy='mean')

    bare_features.transform(['Embarked'], 'fillna', strategy='value', value='S')

    bare_features.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})

    bare_features.transform(['Pclass', 'Embarked'], 'onehot')

    return bare_features



bare_features = add_bare_features(titanic)

titanic.overview()
titanic.plot_correlations(figsize=12, top_n=20)
titanic.grid_search_all()

sns.set_style('whitegrid')

titanic.plot_cv_runs()

compare_last_run_with_real_data(titanic)
k = Kaggle(experiment=titanic, index_feature='PassengerId', target_feature='Survived')

submissions = k.create_submissions(out_folder='bare')

hack_to_fix_submissions('bare')

for submission_name in submissions.columns:

    predictions = submissions[submission_name].values.astype(int)

    print('{}: {:.4f}'.format(submission_name, accuracy_score(real_values, predictions)))
# Define some feature engineering helpers, this is the maximum level of

# flexibility that hallucinate provides for feature engineering. By using

# the 'method' transformation you indicate the method that, applied to your

# entire feature set - including previously defined/transformed features -

# returns the values for the engineered feature



def extract_title(df):

    return df['Name'].apply(lambda x: x.split('.')[0]).apply(lambda x: x.split(' ')[-1])



def extract_surname(df):

    def do_it(a):

        return a[0].split(',')[0] if a[1] > 1 else 'UNK'

    return df[['Name', 'FamilySize']].apply(do_it, axis=1)



def extract_cabin(df):

    return df['Cabin'].apply(lambda x: x[0])



def extract_ticket(df):

    def ret_ticket(ticket):

            ticket = ticket.replace('.', '')

            ticket = ticket.replace('/', '')

            ticket = ticket.split()

            ticket = map(lambda t: t.strip(), ticket)

            ticket = [a for a in filter(lambda t: not t.isdigit(), ticket)]

            if len(ticket) > 0:

                return ticket[0]

            else:

                return 'XXX'

    return df['Ticket'].apply(ret_ticket)



# Let's call them engineered as the work needed for them is a little bit more involved

# Also see that we're inheriting from the 'Bare' ones by setting the 'parent=' during creation

engineered_feature_names = ['FamilySize', 'Title', 'Ticket', 'Cabin', 'Surname']



def add_engineered_features(experiment, existing_features=None, parent_features=None):

    if not existing_features:

        engineered_features = experiment.make_features(engineered_feature_names, name='Engineered features',

                                                    train_data=train_df1,test_data=test_df,

                                                    target='Survived', parent=parent_features)

    else:

        engineered_features = existing_features

        engineered_features.features = engineered_features.features + engineered_feature_names

    engineered_features.transform(['FamilySize'], '+', from_=['SibSp,Parch,1'])

    engineered_features.transform(['Title'], 'method', method_handle=extract_title)

    engineered_features.transform(['Surname'], 'method', method_handle=extract_surname)

    engineered_features.transform(['Ticket'], 'method', from_=['Ticket'], method_handle=extract_ticket)

    engineered_features.transform(['Cabin'], 'fillna', strategy='value', value='_')

    engineered_features.transform(['Cabin'], 'method', method_handle=extract_cabin)

    engineered_features.transform(['Cabin', 'Title', 'Ticket', 'Surname'], 'onehot')

    return engineered_features
engineered_features = add_engineered_features(titanic, parent_features=bare_features)

titanic.grid_search_all()

titanic.plot_cv_runs()

compare_last_run_with_real_data(titanic)
# Let's call them hallucinated features as we're in the near SF domain with some of these

hallucinated_feature_names = ['FareD', 'AgeD', 'Fare^2', 'AgeFareRatio', 'FarePerPerson',

                              'Single', 'SmallFamily', 'LargeFamily', 'AgeFareRatio^2']



def add_hallucinated_features(experiment, existing_features=None, parent_features=None):

    if not existing_features:

        hallucinated_features = experiment.make_features(hallucinated_feature_names, name='Hallucinated features',

                                                    train_data=train_df1,test_data=test_df,

                                                    target='Survived', parent=parent_features)

    else:

        hallucinated_features = existing_features

        hallucinated_features.features = hallucinated_features.features + hallucinated_feature_names

    # Small bug in hallucinate that forces me to re-compute the field in child feature sets

    hallucinated_features.transform(['FamilySize'], '+', from_=['SibSp,Parch,1'])

    hallucinated_features.transform(['Single'], 'map', from_=['FamilySize'], mapping={1: 1, '_others': 0})

    hallucinated_features.transform(['SmallFamily'], 'map', from_=['FamilySize'],

                     mapping={2: 1, 3: 1, 4: 1, '_others': 0})

    hallucinated_features.transform(['LargeFamily'], 'map', from_=['FamilySize'],

                     mapping={1: 0, 2: 0, 3: 0, 4: 0, '_others': 1})

    hallucinated_features.transform(['Age'], 'fillna', strategy='median')

    hallucinated_features.transform(['AgeD'], 'discretize', from_=['Age'], values_range=[0, 19, 25, 35, 60])

    hallucinated_features.transform(['Fare'], 'fillna', strategy='mean')

    hallucinated_features.transform(['FareD'], 'discretize', from_=['Fare'], values_range=[-0.1, 10, 30])

    hallucinated_features.transform(['Fare^2'], '*', from_=['Fare,Fare'])

    hallucinated_features.transform(['TmpFare'], 'fillna', strategy='mean', from_=['Fare'])

    hallucinated_features.transform(['TmpFare'], '+', from_=['TmpFare,0.1'])

    hallucinated_features.transform(['AgeFareRatio'], '/', from_=['Age,TmpFare'])

    hallucinated_features.transform(['AgeFareRatio^2'], '*', from_=['AgeFareRatio,AgeFareRatio'])

    hallucinated_features.transform(['FarePerPerson'], '/', from_=['TmpFare,FamilySize'])

    return hallucinated_features
add_hallucinated_features(titanic, parent_features=engineered_features)

titanic.grid_search_all()

titanic.plot_cv_runs()

compare_last_run_with_real_data(titanic)
titanic = Experiment(name='Titanic', cv=30, cv_shuffle=False, sc='accuracy', parallel=False)

titanic.add_estimator(EstimatorConfig(LGBMClassifier(), {}, 'LGB'))

titanic.add_estimator(EstimatorConfig(XGBClassifier(), {}, 'XGB'))

titanic.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LRE'))



titanic_features = add_bare_features(titanic)

titanic_features.name = "All features"

add_engineered_features(titanic, existing_features=titanic_features)

add_hallucinated_features(titanic, existing_features=titanic_features)



titanic.overview()

titanic.grid_search_all()

compare_last_run_with_real_data(titanic)

titanic.plot_cv_runs()
titanic.plot_feature_importance(figsize=10)
k = Kaggle(experiment=titanic, index_feature='PassengerId', target_feature='Survived')

submissions = k.create_submissions(out_folder='hal_features')

hack_to_fix_submissions('hal_features')

for submission_name in submissions.columns:

    predictions = submissions[submission_name].values.astype(int)

    print('{}: {:.4f}'.format(submission_name, accuracy_score(real_values, predictions)))
lb_overfit = Experiment(name="LB Overfit", cv=30, cv_shuffle=False, sc='accuracy', parallel=False)



features = ['Pclass', 'Sex', 'Fare', 'FareD', 'Fare^2', 'AgeFareRatio', 'FarePerPerson',

            'AgeD', 'FamilySize', 'SibSp', 'Parch', 'Single', 'SmallFamily', 'LargeFamily',

            'Cabin', 'Embarked', 'Title', 'Surname', 'Ticket', 'AgeFareRatio^2']



t1 = lb_overfit.make_features(features, name='Overfit Features',

                            train_data=train_df, test_data=test_df, target='Survived')

    

t1.set_feature_selector(SelectFromModel(

    DecisionTreeClassifier(random_state=7, max_depth=12), threshold=0.0005))



t1.transform(['Fare'], 'fillna', strategy='mean')



t1.transform(['Pclass'], 'onehot')

t1.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})



t1.transform(['FareD'], 'discretize', from_=['Fare'], values_range=[-0.1, 10, 30])

t1.transform(['FareD'], 'onehot')



t1.transform(['Fare^2'], '*', from_=['Fare, Fare'])



# TODO this will look much better with an ExpressionTransform(expr='SibSp + Parch + 1')

t1.transform(['FamilySize'], '+', from_=['SibSp,Parch,1'])

t1.transform(['Single'], 'map', from_=['FamilySize'], mapping={1: 1, '_others': 0})

t1.transform(['SmallFamily'], 'map', from_=['FamilySize'],

             mapping={2: 1, 3: 1, 4: 1, '_others': 0})

t1.transform(['LargeFamily'], 'map', from_=['FamilySize'],

             mapping={1: 0, 2: 0, 3: 0, 4: 0, '_others': 1})



t1.transform(['Cabin'], 'fillna', strategy='value', value='_')

t1.transform(['Cabin'], 'method', method_handle=extract_cabin)

t1.transform(['Cabin'], 'onehot')



t1.transform(['Title'], 'method', from_=['Name'], method_handle=extract_title)

t1.transform(['Title'], 'onehot')



t1.transform(['FarePerPerson'], '/', from_=['Fare,FamilySize'])

t1.transform(['FarePerPerson^2'], '*', from_=['FarePerPerson,FarePerPerson'])



t1.transform(['Surname'], 'method', from_=['Name,FamilySize'],

             method_handle=extract_surname)

t1.transform(['Surname'], 'onehot')



t1.transform(['Ticket'], 'method', from_=['Ticket'],

             method_handle=extract_ticket)

t1.transform(['Ticket'], 'onehot')



t1.transform(['Embarked'], 'fillna', strategy='value', value='C')

t1.transform(['Embarked'], 'onehot')



t1.transform(['Age'], 'fillna', strategy='median')



t1.transform(['TmpFare'], 'fillna', from_=['Fare'], strategy='mean')

t1.transform(['TmpFare'], '+', from_=['TmpFare,0.1'])

t1.transform(['AgeFareRatio'], '/', from_=['Age,TmpFare'])

t1.transform(['AgeFareRatio^2'], '*', from_=['AgeFareRatio,AgeFareRatio'])



t1.transform(['AgeD'], 'discretize', from_=['Age'], values_range=[0, 6, 13, 19, 25, 35, 60])

t1.transform(['AgeD'], 'onehot')



# exp.add_config(EstimatorConfig(DecisionTreeClassifier(), {}, 'DTR'))

# lb_overfit.add_estimator(EstimatorConfig(XGBClassifier(nthread=1), {}, 'XGB'))

lb_overfit.add_estimator(EstimatorConfig(LGBMClassifier(nthread=1), {}, 'LGB'))

# lb_overfit.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LR'))

# exp.add_config(EstimatorConfig(KNeighborsClassifier(n_jobs=1), {}, 'KNN'))

# exp.add_config(EstimatorConfig(VotingBuilder(exp.non_stacking_configs()), {}, 'VOT',

#                                 stacking=True))



dtr_f_sel_thresholds = [0.0005 + a * 0.0005 for a in range(5)]

lb_overfit.grid_search_all(f_sel_thresholds=dtr_f_sel_thresholds)
lb_overfit.plot_cv_runs()

lb_overfit.plot_f_sel_learning_curve()
k = Kaggle(experiment=lb_overfit, index_feature='PassengerId', target_feature='Survived')

submissions = k.create_submissions(out_folder='overfit_lb', verbose=True)

hack_to_fix_submissions('overfit_lb')

for submission_name in submissions.columns:

    predictions = submissions[submission_name].values.astype(int)

    print('{}: {:.4f}'.format(submission_name, accuracy_score(real_values, predictions)))
import folium

m = folium.Map(

    location=[52.48, -1.9025],

    zoom_start=7,

)

#PremierLeague

logo_urlchelsea = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/cc/Chelsea_FC.svg/800px-Chelsea_FC.svg.png'

escudochelsea = folium.features.CustomIcon(logo_urlchelsea,icon_size=(40,40))

folium.Marker([51.481667, -0.191111], popup='<br><a href= https://en.wikipedia.org/wiki/Chelsea_F.C. >https://en.wikipedia.org/wiki/Chelsea_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 41841'+'<br><b>Official Web:</b> https://www.chelseafc.com/en', tooltip='<b>Stamford Bridge - CHELSEA</b>', icon=escudochelsea).add_to(m)

logo_urlbrighton = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/fd/Brighton_%26_Hove_Albion_logo.svg/200px-Brighton_%26_Hove_Albion_logo.svg.png'

escudobrighton = folium.features.CustomIcon(logo_urlbrighton,icon_size=(50,50))

folium.Marker([50.861822, -0.083278], popup='<br><a href= https://en.wikipedia.org/wiki/Brighton_%26_Hove_Albion_F.C. >https://en.wikipedia.org/wiki/Brighton_%26_Hove_Albion_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 30750'+'<br><b>Official Web:</b> https://www.brightonandhovealbion.com/', tooltip='<b>Amex Stadium - BRIGHTON</b>', icon=escudobrighton).add_to(m)

logo_urlwatford = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/e2/Watford.svg/190px-Watford.svg.png'

escudowatford = folium.features.CustomIcon(logo_urlwatford,icon_size=(40,40))

folium.Marker([51.649836, -0.401486], popup='<br><a href= https://en.wikipedia.org/wiki/Watford_F.C. >https://en.wikipedia.org/wiki/Watford_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 21577'+'<br><b>Official Web:</b> https://www.watfordfc.com/', tooltip='<b>Vicarage Road - WATFORD</b>', icon=escudowatford).add_to(m)

logo_urlpalace = 'https://upload.wikimedia.org/wikipedia/en/thumb/0/0c/Crystal_Palace_FC_logo.svg/170px-Crystal_Palace_FC_logo.svg.png'

escudopalace = folium.features.CustomIcon(logo_urlpalace,icon_size=(40,40))

folium.Marker([51.398333, -0.085556], popup='<br><a href= https://en.wikipedia.org/wiki/Crystal_Palace_F.C. >https://en.wikipedia.org/wiki/Crystal_Palace_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 25486'+'<br><b>Official Web:</b> https://www.cpfc.co.uk/', tooltip='<b>Selhurst Park - CRYSTAL PALACE</b>', icon=escudopalace).add_to(m)

logo_urltottenham = 'https://upload.wikimedia.org/wikipedia/en/thumb/b/b4/Tottenham_Hotspur.svg/110px-Tottenham_Hotspur.svg.png'

escudotottenham = folium.features.CustomIcon(logo_urltottenham,icon_size=(40,40))

folium.Marker([51.604444, -0.066389], popup='<br><a href= https://en.wikipedia.org/wiki/Tottenham_Hotspur_F.C. >https://en.wikipedia.org/wiki/Tottenham_Hotspur_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 62303'+'<br><b>Official Web:</b> https://www.tottenhamhotspur.com/', tooltip='<b>Tottenham Hotspur Stadium - TOTTENHAM</b>', icon=escudotottenham).add_to(m)

logo_urlmanunited = 'https://upload.wikimedia.org/wikipedia/ca/thumb/7/7a/Manchester_United_FC_crest.svg/1200px-Manchester_United_FC_crest.svg.png'

escudomanunited = folium.features.CustomIcon(logo_urlmanunited,icon_size=(40,40))

folium.Marker([53.463056, -2.291389], popup='<br><a href= https://en.wikipedia.org/wiki/Manchester_United_F.C. >https://en.wikipedia.org/wiki/Manchester_United_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 74879'+'<br><b>Official Web:</b> https://www.manutd.com/', tooltip='<b>Old Trafford - MANCHESTER UNITED</b>', icon=escudomanunited).add_to(m)

logo_urlmancity = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Manchester_City_FC_badge.svg/800px-Manchester_City_FC_badge.svg.png'

escudomancity = folium.features.CustomIcon(logo_urlmancity,icon_size=(40,40))

folium.Marker([53.483056, -2.200278], popup='<br><a href= https://en.wikipedia.org/wiki/Manchester_City_F.C. >https://en.wikipedia.org/wiki/Manchester_City_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 55097'+'<br><b>Official Web:</b> https://www.mancity.com/', tooltip='<b>Etihad Stadium - MANCHESTER CITY</b>', icon=escudomancity).add_to(m)

logo_urleverton = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/7c/Everton_FC_logo.svg/1200px-Everton_FC_logo.svg.png'

escudoeverton = folium.features.CustomIcon(logo_urleverton,icon_size=(40,40))

folium.Marker([53.438889, -2.966389], popup='<br><a href= https://en.wikipedia.org/wiki/Everton_F.C. >https://en.wikipedia.org/wiki/Everton_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 39572'+'<br><b>Official Web:</b> https://www.evertonfc.com/', tooltip='<b>Goodison Park - EVERTON</b>', icon=escudoeverton).add_to(m)

logo_urlliverpool = 'https://upload.wikimedia.org/wikipedia/en/thumb/0/0c/Liverpool_FC.svg/180px-Liverpool_FC.svg.png'

escudoliverpool = folium.features.CustomIcon(logo_urlliverpool,icon_size=(40,50))

folium.Marker([53.430833, -2.960833], popup='<br><a href= https://en.wikipedia.org/wiki/Liverpool_F.C. >https://en.wikipedia.org/wiki/Liverpool_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 53394'+'<br><b>Official Web:</b> https://www.liverpoolfc.com/', tooltip='<b>Anfield - LIVERPOOL</b>', icon=escudoliverpool).add_to(m)

logo_urlwolves = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/fc/Wolverhampton_Wanderers.svg/800px-Wolverhampton_Wanderers.svg.png'

escudowolves = folium.features.CustomIcon(logo_urlwolves,icon_size=(40,40))

folium.Marker([52.590278, -2.130278], popup='<br><a href= https://en.wikipedia.org/wiki/Wolverhampton_Wanderers_F.C. >https://en.wikipedia.org/wiki/Wolverhampton_Wanderers_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 32050'+'<br><b>Official Web:</b> https://www.wolves.co.uk/', tooltip='<b>Molineux Stadium - WOLVERHAMPTON</b>', icon=escudowolves).add_to(m)

logo_urlleicester = 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2d/Leicester_City_crest.svg/800px-Leicester_City_crest.svg.png'

escudoleicester = folium.features.CustomIcon(logo_urlleicester,icon_size=(40,40))

folium.Marker([52.620278, -1.142222], popup='<br><a href= https://en.wikipedia.org/wiki/Leicester_City_F.C. >https://en.wikipedia.org/wiki/Leicester_City_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 32261'+'<br><b>Official Web:</b> https://www.lcfc.com/', tooltip='<b>King Power Stadium - LEICESTER CITY</b>', icon=escudoleicester).add_to(m)

logo_urlwestham = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c2/West_Ham_United_FC_logo.svg/185px-West_Ham_United_FC_logo.svg.png'

escudowestham = folium.features.CustomIcon(logo_urlwestham,icon_size=(40,40))

folium.Marker([51.538611, -0.016389], popup='<br><a href= https://en.wikipedia.org/wiki/West_Ham_United_F.C. >https://en.wikipedia.org/wiki/West_Ham_United_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 60000'+'<br><b>Official Web:</b> https://www.whufc.com/', tooltip='<b>London Stadium - WEST HAM UNITED</b>', icon=escudowestham).add_to(m)

logo_urlsheffield = 'https://upload.wikimedia.org/wikipedia/en/thumb/9/9c/Sheffield_United_FC_logo.svg/180px-Sheffield_United_FC_logo.svg.png'

escudosheffield = folium.features.CustomIcon(logo_urlsheffield,icon_size=(40,40))

folium.Marker([53.370278, -1.470833], popup='<br><a href= https://en.wikipedia.org/wiki/Sheffield_United_F.C. >https://en.wikipedia.org/wiki/Sheffield_United_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 32050'+'<br><b>Official Web:</b> https://www.sufc.co.uk/', tooltip='<b>Bramall Lane - SHEFFIELD UNITED</b>', icon=escudosheffield).add_to(m)

logo_urlarsenal = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Arsenal_FC.svg/170px-Arsenal_FC.svg.png'

escudoarsenal = folium.features.CustomIcon(logo_urlarsenal,icon_size=(40,40))

folium.Marker([51.555, -0.108611], popup='<br><a href= https://en.wikipedia.org/wiki/Arsenal_F.C. >https://en.wikipedia.org/wiki/Arsenal_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 60704'+'<br><b>Official Web:</b> https://www.arsenal.com/', tooltip='<b>Emirates Stadium - ARSENAL</b>', icon=escudoarsenal).add_to(m)

logo_urlsouthampton = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c9/FC_Southampton.svg/180px-FC_Southampton.svg.png'

escudosouthampton = folium.features.CustomIcon(logo_urlsouthampton,icon_size=(40,40))

folium.Marker([50.905833, -1.391111], popup='<br><a href= https://en.wikipedia.org/wiki/Southampton_F.C. >https://en.wikipedia.org/wiki/Southampton_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 32505'+'<br><b>Official Web:</b> https://www.southamptonfc.com/', tooltip='<b>St Marys Stadium - SOUTHAMPTON</b>', icon=escudosouthampton).add_to(m)

logo_urlnewcastle = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Newcastle_United_Logo.svg/200px-Newcastle_United_Logo.svg.png'

escudonewcastle = folium.features.CustomIcon(logo_urlnewcastle,icon_size=(40,40))

folium.Marker([54.975556, -1.621667], popup='<br><a href= https://en.wikipedia.org/wiki/Newcastle_United_F.C. >https://en.wikipedia.org/wiki/Newcastle_United_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 52305'+'<br><b>Official Web:</b> https://www.nufc.co.uk/', tooltip='<b>St James Park - NEWCASTLE</b>', icon=escudonewcastle).add_to(m)

logo_urlburnley = 'https://upload.wikimedia.org/wikipedia/en/thumb/6/62/Burnley_F.C._Logo.svg/180px-Burnley_F.C._Logo.svg.png'

escudoburnley = folium.features.CustomIcon(logo_urlburnley,icon_size=(40,40))

folium.Marker([53.789167, -2.230278], popup='<br><a href= https://en.wikipedia.org/wiki/Burnley_F.C. >https://en.wikipedia.org/wiki/Burnley_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 21944'+'<br><b>Official Web:</b> https://www.burnleyfootballclub.com/', tooltip='<b>Turf Moor - BURNLEY</b>', icon=escudoburnley).add_to(m)

logo_urlbournemouth = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/AFC_Bournemouth_%282013%29.svg/170px-AFC_Bournemouth_%282013%29.svg.png'

escudobournemouth = folium.features.CustomIcon(logo_urlbournemouth,icon_size=(40,40))

folium.Marker([50.735278, -1.838333], popup='<br><a href= https://en.wikipedia.org/wiki/A.F.C._Bournemouth >https://en.wikipedia.org/wiki/A.F.C._Bournemouth</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 11364'+'<br><b>Official Web:</b> https://www.afcb.co.uk/', tooltip='<b>Vitality Stadium - BOURNEMOUTH</b>', icon=escudobournemouth).add_to(m)

logo_urlvilla = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f9/Aston_Villa_FC_crest_%282016%29.svg/150px-Aston_Villa_FC_crest_%282016%29.svg.png'

escudovilla = folium.features.CustomIcon(logo_urlvilla,icon_size=(40,40))

folium.Marker([52.509167, -1.884722], popup='<br><a href= https://en.wikipedia.org/wiki/Aston_Villa_F.C. >https://en.wikipedia.org/wiki/Aston_Villa_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 42095'+'<br><b>Official Web:</b> https://www.avfc.co.uk/', tooltip='<b>Villa Park - ASTON VILLA</b>', icon=escudovilla).add_to(m)

logo_urlnorwich = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8c/Norwich_City.svg/150px-Norwich_City.svg.png'

escudonorwich = folium.features.CustomIcon(logo_urlnorwich,icon_size=(40,40))

folium.Marker([52.622128, 1.308653], popup='<br><a href= https://en.wikipedia.org/wiki/Norwich_City_F.C. >https://en.wikipedia.org/wiki/Norwich_City_F.C.</a>'+'<br><b>Division:</b> Premier League'+'<br><b>Capacity:</b> 27244'+'<br><b>Official Web:</b> https://www.canaries.co.uk/', tooltip='<b>Carrow Road - NORWICH CITY</b>', icon=escudonorwich).add_to(m)

#Championship

logo_urlleeds = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/54/Leeds_United_F.C._logo.svg/165px-Leeds_United_F.C._logo.svg.png'

escudoleeds = folium.features.CustomIcon(logo_urlleeds,icon_size=(40,40))

folium.Marker([53.777778, -1.572222], popup='<br><a href= https://en.wikipedia.org/wiki/Leeds_United_F.C. >https://en.wikipedia.org/wiki/Leeds_United_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 37890'+'<br><b>Official Web:</b> https://www.leedsunited.com/', tooltip='<b>Elland Road - LEEDS UNITED</b>', icon=escudoleeds).add_to(m)

logo_urlwestbrom = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/West_Bromwich_Albion.svg/160px-West_Bromwich_Albion.svg.png'

escudowestbrom = folium.features.CustomIcon(logo_urlwestbrom,icon_size=(40,40))

folium.Marker([52.509167, -1.963889], popup='<br><a href= https://en.wikipedia.org/wiki/West_Bromwich_Albion_F.C. >https://en.wikipedia.org/wiki/West_Bromwich_Albion_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 26688'+'<br><b>Official Web:</b> https://www.wba.co.uk/', tooltip='<b>The Hawthorns - WEST BROMWICH ALBION</b>', icon=escudowestbrom).add_to(m)

logo_urlfulham = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Fulham_FC_%28shield%29.svg/150px-Fulham_FC_%28shield%29.svg.png'

escudofulham = folium.features.CustomIcon(logo_urlfulham,icon_size=(40,40))

folium.Marker([51.475, -0.221667], popup='<br><a href= https://en.wikipedia.org/wiki/Fulham_F.C. >https://en.wikipedia.org/wiki/Fulham_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 25700'+'<br><b>Official Web:</b> https://www.fulhamfc.com/', tooltip='<b>Craven Cottage - FULHAM</b>', icon=escudofulham).add_to(m)

logo_urlbrentford = 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/Brentford_FC_crest.svg/180px-Brentford_FC_crest.svg.png'

escudobrentford = folium.features.CustomIcon(logo_urlbrentford,icon_size=(40,40))

folium.Marker([51.488183, -0.302639], popup='<br><a href= https://en.wikipedia.org/wiki/Brentford_F.C. >https://en.wikipedia.org/wiki/Brentford_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 12300'+'<br><b>Official Web:</b> https://www.brentfordfc.com/', tooltip='<b>Griffin Park - BRENTFORD</b>', icon=escudobrentford).add_to(m)

logo_urlnottingham = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/Nottingham_Forest_F.C._logo.svg/100px-Nottingham_Forest_F.C._logo.svg.png'

escudonottingham = folium.features.CustomIcon(logo_urlnottingham,icon_size=(40,40))

folium.Marker([52.94, -1.132778], popup='<br><a href= https://en.wikipedia.org/wiki/Nottingham_Forest_F.C. >https://en.wikipedia.org/wiki/Nottingham_Forest_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 30445'+'<br><b>Official Web:</b> https://www.nottinghamforest.co.uk/', tooltip='<b>City Ground - NOTTINGHAM FOREST</b>', icon=escudonottingham).add_to(m)

logo_urlcardiff = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/3c/Cardiff_City_crest.svg/200px-Cardiff_City_crest.svg.png'

escudocardiff = folium.features.CustomIcon(logo_urlcardiff,icon_size=(40,40))

folium.Marker([51.472778, -3.203056], popup='<br><a href= https://en.wikipedia.org/wiki/Cardiff_City_F.C. >https://en.wikipedia.org/wiki/Cardiff_City_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 33280'+'<br><b>Official Web:</b> https://www.cardiffcityfc.co.uk/', tooltip='<b>Cardiff City Stadium - CARDIFF CITY</b>', icon=escudocardiff).add_to(m)

logo_urlmillwall = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c9/Millwall_F.C._logo.svg/200px-Millwall_F.C._logo.svg.png'

escudomillwall = folium.features.CustomIcon(logo_urlmillwall,icon_size=(40,40))

folium.Marker([51.485953, -0.05095], popup='<br><a href= https://en.wikipedia.org/wiki/Millwall_F.C. >https://en.wikipedia.org/wiki/Millwall_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 20146'+'<br><b>Official Web:</b> https://www.millwallfc.co.uk/', tooltip='<b>The Den - MILLWALL</b>', icon=escudomillwall).add_to(m)

logo_urlswansea = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f9/Swansea_City_AFC_logo.svg/220px-Swansea_City_AFC_logo.svg.png'

escudoswansea = folium.features.CustomIcon(logo_urlswansea,icon_size=(40,40))

folium.Marker([51.6422, -3.9351], popup='<br><a href= https://en.wikipedia.org/wiki/Swansea_City_A.F.C. >https://en.wikipedia.org/wiki/Swansea_City_A.F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 21088'+'<br><b>Official Web:</b> https://www.swanseacity.com/', tooltip='<b>Liberty Stadium - SWANSEA CITY</b>', icon=escudoswansea).add_to(m)

logo_urlpreston = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Preston_North_End_FC.svg/180px-Preston_North_End_FC.svg.png'

escudopreston = folium.features.CustomIcon(logo_urlpreston,icon_size=(40,40))

folium.Marker([53.772222, -2.688056], popup='<br><a href= https://en.wikipedia.org/wiki/Preston_North_End_F.C. >https://en.wikipedia.org/wiki/Preston_North_End_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 23404'+'<br><b>Official Web:</b> https://www.pnefc.net/', tooltip='<b>Deepdale Stadium - PRESTON NORTH END</b>', icon=escudopreston).add_to(m)

logo_urlbristol = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f5/Bristol_City_crest.svg/200px-Bristol_City_crest.svg.png'

escudobristol = folium.features.CustomIcon(logo_urlbristol,icon_size=(40,40))

folium.Marker([51.44, -2.620278], popup='<br><a href= https://en.wikipedia.org/wiki/Bristol_City_F.C. >https://en.wikipedia.org/wiki/Bristol_City_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 27000'+'<br><b>Official Web:</b> https://www.bcfc.co.uk/', tooltip='<b>Ashton Gate - BRISTOL CITY</b>', icon=escudobristol).add_to(m)

logo_urlderby = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4a/Derby_County_crest.svg/180px-Derby_County_crest.svg.png'

escudoderby = folium.features.CustomIcon(logo_urlderby,icon_size=(40,40))

folium.Marker([52.915, -1.447222], popup='<br><a href= https://en.wikipedia.org/wiki/Derby_County_F.C. >https://en.wikipedia.org/wiki/Derby_County_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 33597'+'<br><b>Official Web:</b> https://www.dcfc.co.uk/', tooltip='<b>Pride Park Stadium - DERBY COUNTY</b>', icon=escudoderby).add_to(m)

logo_urlblackburn = 'https://upload.wikimedia.org/wikipedia/en/thumb/0/0f/Blackburn_Rovers.svg/180px-Blackburn_Rovers.svg.png'

escudoblackburn = folium.features.CustomIcon(logo_urlblackburn,icon_size=(40,40))

folium.Marker([53.728611, -2.489167], popup='<br><a href= https://en.wikipedia.org/wiki/Blackburn_Rovers_F.C. >https://en.wikipedia.org/wiki/Blackburn_Rovers_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 31367'+'<br><b>Official Web:</b> https://www.rovers.co.uk/', tooltip='<b>Ewood Park - BLACKBURN ROVERS</b>', icon=escudoblackburn).add_to(m)

logo_urlwigan = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/43/Wigan_Athletic.svg/220px-Wigan_Athletic.svg.png'

escudowigan = folium.features.CustomIcon(logo_urlwigan,icon_size=(40,40))

folium.Marker([53.547778, -2.653889], popup='<br><a href= https://en.wikipedia.org/wiki/Wigan_Athletic_F.C. >https://en.wikipedia.org/wiki/Wigan_Athletic_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 25138'+'<br><b>Official Web:</b> https://wiganathletic.com/', tooltip='<b>DW Stadium - WIGAN ATHLETIC</b>', icon=escudowigan).add_to(m)

logo_urlreading = 'https://upload.wikimedia.org/wikipedia/en/thumb/1/11/Reading_FC.svg/200px-Reading_FC.svg.png'

escudoreading = folium.features.CustomIcon(logo_urlreading,icon_size=(40,40))

folium.Marker([51.422222, -0.982778], popup='<br><a href= https://en.wikipedia.org/wiki/Reading_F.C. >https://en.wikipedia.org/wiki/Reading_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 24161'+'<br><b>Official Web:</b> https://www.readingfc.co.uk/', tooltip='<b>Madjeski Stadium - READING FC</b>', icon=escudoreading).add_to(m)

logo_urlwednesday = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/88/Sheffield_Wednesday_badge.svg/150px-Sheffield_Wednesday_badge.svg.png'

escudowednesday = folium.features.CustomIcon(logo_urlwednesday,icon_size=(40,40))

folium.Marker([53.411389, -1.500556], popup='<br><a href= https://en.wikipedia.org/wiki/Sheffield_Wednesday_F.C. >https://en.wikipedia.org/wiki/Sheffield_Wednesday_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 39732'+'<br><b>Official Web:</b> https://www.swfc.co.uk//', tooltip='<b>Hillsborough Stadium - SHEFFIELD WEDNESDAY</b>', icon=escudowednesday).add_to(m)

logo_urlqpr = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/31/Queens_Park_Rangers_crest.svg/180px-Queens_Park_Rangers_crest.svg.png'

escudoqpr = folium.features.CustomIcon(logo_urlqpr,icon_size=(40,40))

folium.Marker([51.509167, -0.232222], popup='<br><a href= https://en.wikipedia.org/wiki/Queens_Park_Rangers_F.C. >https://en.wikipedia.org/wiki/Queens_Park_Rangers_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 18439'+'<br><b>Official Web:</b> https://www.qpr.co.uk/', tooltip='<b>Loftus Road - QUEENS PARK RANGERS</b>', icon=escudoqpr).add_to(m)

logo_urlstoke = 'https://upload.wikimedia.org/wikipedia/en/thumb/2/29/Stoke_City_FC.svg/220px-Stoke_City_FC.svg.png'

escudostoke = folium.features.CustomIcon(logo_urlstoke,icon_size=(40,40))

folium.Marker([52.988333, -2.175556], popup='<br><a href= https://en.wikipedia.org/wiki/Stoke_City_F.C. >https://en.wikipedia.org/wiki/Stoke_City_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 30089'+'<br><b>Official Web:</b> https://www.stokecityfc.com/', tooltip='<b>Bet 365 Stadium - STOKE CITY</b>', icon=escudostoke).add_to(m)

logo_urlmiddlesbrough = 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2c/Middlesbrough_FC_crest.svg/200px-Middlesbrough_FC_crest.svg.png'

escudomiddlesbrough = folium.features.CustomIcon(logo_urlmiddlesbrough,icon_size=(40,40))

folium.Marker([54.578333, -1.216944], popup='<br><a href= https://en.wikipedia.org/wiki/Middlesbrough_F.C. >https://en.wikipedia.org/wiki/Middlesbrough_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 34742'+'<br><b>Official Web:</b> https://www.mfc.co.uk/', tooltip='<b>Riverside Stadium - MIDDLESBROUGH</b>', icon=escudomiddlesbrough).add_to(m)

logo_urlbirmingham = 'https://upload.wikimedia.org/wikipedia/en/thumb/6/68/Birmingham_City_FC_logo.svg/100px-Birmingham_City_FC_logo.svg.png'

escudobirmingham = folium.features.CustomIcon(logo_urlbirmingham,icon_size=(50,50))

folium.Marker([52.475000, -1.86700], popup='<br><a href= https://en.wikipedia.org/wiki/Birmingham_City_F.C. >https://en.wikipedia.org/wiki/Birmingham_City_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 29409'+'<br><b>Official Web:</b> https://www.bcfc.com/', tooltip='<b>St. Andrews Stadium - BIRMINGHAM CITY</b>', icon=escudobirmingham).add_to(m)

logo_urlhuddersfield = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Huddersfield_Town_A.F.C._logo.png/150px-Huddersfield_Town_A.F.C._logo.png'

escudohuddersfield = folium.features.CustomIcon(logo_urlhuddersfield,icon_size=(40,40))

folium.Marker([53.654167, -1.768333], popup='<br><a href= https://en.wikipedia.org/wiki/Huddersfield_Town_A.F.C. >https://en.wikipedia.org/wiki/Huddersfield_Town_A.F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 24121'+'<br><b>Official Web:</b> https://www.htafc.com/', tooltip='<b>John Smiths Stadium - HUDDERSFIELD TOWN</b>', icon=escudohuddersfield).add_to(m)

logo_urlcharlton = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/CharltonBadge_30Jan2020.png/200px-CharltonBadge_30Jan2020.png'

escudocharlton = folium.features.CustomIcon(logo_urlcharlton,icon_size=(40,40))

folium.Marker([51.486389, 0.036389], popup='<br><a href= https://en.wikipedia.org/wiki/Charlton_Athletic_F.C. >https://en.wikipedia.org/wiki/Charlton_Athletic_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 27111'+'<br><b>Official Web:</b> https://www.cafc.co.uk/', tooltip='<b>The Valley - CHARLTON ATHLETIC</b>', icon=escudocharlton).add_to(m)

logo_urlhull = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/54/Hull_City_A.F.C._logo.svg/150px-Hull_City_A.F.C._logo.svg.png'

escudohull = folium.features.CustomIcon(logo_urlhull,icon_size=(40,40))

folium.Marker([53.746111, -0.367778], popup='<br><a href= https://en.wikipedia.org/wiki/Hull_City_A.F.C. >https://en.wikipedia.org/wiki/Hull_City_A.F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 25400'+'<br><b>Official Web:</b> https://www.hullcitytigers.com/', tooltip='<b>KCOM Stadium - HULL CITY</b>', icon=escudohull).add_to(m)

logo_urlluton = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/LutonTownFC2009.png/180px-LutonTownFC2009.png'

escudoluton = folium.features.CustomIcon(logo_urlluton,icon_size=(40,40))

folium.Marker([51.884167, -0.431667], popup='<br><a href= https://en.wikipedia.org/wiki/Luton_Town_F.C. >https://en.wikipedia.org/wiki/Luton_Town_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 10356'+'<br><b>Official Web:</b> https://www.lutontown.co.uk/', tooltip='<b>Kenilworth Road - LUTON TOWN</b>', icon=escudoluton).add_to(m)

logo_urlbarnsley = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c9/Barnsley_FC.svg/200px-Barnsley_FC.svg.png'

escudobarnsley = folium.features.CustomIcon(logo_urlbarnsley,icon_size=(40,40))

folium.Marker([53.552222, -1.4675], popup='<br><a href= https://en.wikipedia.org/wiki/Barnsley_F.C. >https://en.wikipedia.org/wiki/Barnsley_F.C.</a>'+'<br><b>Division:</b> Championship'+'<br><b>Capacity:</b> 23287'+'<br><b>Official Web:</b> https://www.barnsleyfc.co.uk/', tooltip='<b>Oakwell - BARNSLEY FC</b>', icon=escudobarnsley).add_to(m)

#LeagueOne

logo_urlcoventry = 'https://upload.wikimedia.org/wikipedia/en/thumb/9/94/Coventry_City_FC_logo.svg/150px-Coventry_City_FC_logo.svg.png'

escudocoventry = folium.features.CustomIcon(logo_urlcoventry,icon_size=(40,40))

folium.Marker([52.475703, -1.868189], popup='<br><a href= https://en.wikipedia.org/wiki/Coventry_City_F.C. >https://en.wikipedia.org/wiki/Coventry_City_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 29409'+'<br><b>Official Web:</b> https://www.ccfc.co.uk/', tooltip='<b>St. Andrews Stadium - COVENTRY CITY <br>(on loan Birmingham City Stadium temporarily)</b>', icon=escudocoventry).add_to(m)

logo_urlrotherham = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c0/Rotherham_United_FC.svg/150px-Rotherham_United_FC.svg.png'

escudorotherham = folium.features.CustomIcon(logo_urlrotherham,icon_size=(40,40))

folium.Marker([53.4270, -1.362], popup='<br><a href= https://en.wikipedia.org/wiki/Rotherham_United_F.C. >https://en.wikipedia.org/wiki/Rotherham_United_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 12021'+'<br><b>Official Web:</b> https://www.themillers.co.uk/', tooltip='<b>The AESSEAL New York Stadium - ROTHERHAM UNITED FC</b>', icon=escudorotherham).add_to(m)

logo_urlwycombe = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/fb/Wycombe_Wanderers_FC_logo.svg/200px-Wycombe_Wanderers_FC_logo.svg.png'

escudowycombe = folium.features.CustomIcon(logo_urlwycombe,icon_size=(40,40))

folium.Marker([51.630556, -0.800278], popup='<br><a href= https://en.wikipedia.org/wiki/Wycombe_Wanderers_F.C. >https://en.wikipedia.org/wiki/Wycombe_Wanderers_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 10137'+'<br><b>Official Web:</b> https://www.wycombewanderers.co.uk/', tooltip='<b>Adams Park - WYCOMBE WANDERERS</b>', icon=escudowycombe).add_to(m)

logo_urloxford = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/3e/Oxford_United_FC_logo.svg/150px-Oxford_United_FC_logo.svg.png'

escudooxford = folium.features.CustomIcon(logo_urloxford,icon_size=(40,40))

folium.Marker([51.716419, -1.208067], popup='<br><a href= https://en.wikipedia.org/wiki/Oxford_United_F.C. >https://en.wikipedia.org/wiki/Oxford_United_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 12400'+'<br><b>Official Web:</b> https://www.oufc.co.uk/', tooltip='<b>Kassam Stadium - OXFORD UNITED</b>', icon=escudooxford).add_to(m)

logo_urlportsmouth = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/38/Portsmouth_FC_logo.svg/220px-Portsmouth_FC_logo.svg.png'

escudoportsmouth = folium.features.CustomIcon(logo_urlportsmouth,icon_size=(40,40))

folium.Marker([50.796389, -1.063889], popup='<br><a href= https://en.wikipedia.org/wiki/Portsmouth_F.C. >https://en.wikipedia.org/wiki/Portsmouth_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 20620'+'<br><b>Official Web:</b> https://www.portsmouthfc.co.uk/', tooltip='<b>Fratton Park - PORTSMOUTH FC</b>', icon=escudoportsmouth).add_to(m)

logo_urlfleetwood = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/ed/Fleetwood_Town_F.C._logo.svg/150px-Fleetwood_Town_F.C._logo.svg.png'

escudofleetwood = folium.features.CustomIcon(logo_urlfleetwood,icon_size=(40,40))

folium.Marker([53.9165, -3.0247], popup='<br><a href= https://en.wikipedia.org/wiki/Fleetwood_Town_F.C. >https://en.wikipedia.org/wiki/Fleetwood_Town_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 5327'+'<br><b>Official Web:</b> https://www.fleetwoodtownfc.com/', tooltip='<b>Highbury Stadium - FLEETWOOD TOWN</b>', icon=escudofleetwood).add_to(m)

logo_urlpeterborough = 'https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Peterborough_United.svg/150px-Peterborough_United.svg.png'

escudopeterborough = folium.features.CustomIcon(logo_urlpeterborough,icon_size=(40,40))

folium.Marker([52.564697, -0.240406], popup='<br><a href= https://en.wikipedia.org/wiki/Peterborough_United_F.C. >https://en.wikipedia.org/wiki/Peterborough_United_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 15314'+'<br><b>Official Web:</b> https://www.theposh.com/', tooltip='<b>London Road Stadium - PETERBOROUGH UNITED</b>', icon=escudopeterborough).add_to(m)

logo_urlsunderland = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Logo_Sunderland.svg/180px-Logo_Sunderland.svg.png'

escudosunderland = folium.features.CustomIcon(logo_urlsunderland,icon_size=(40,40))

folium.Marker([54.9144, -1.3882], popup='<br><a href= https://en.wikipedia.org/wiki/Sunderland_A.F.C. >https://en.wikipedia.org/wiki/Sunderland_A.F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 49000'+'<br><b>Official Web:</b> https://www.safc.com/', tooltip='<b>Stadium of Light - SUNDERLAND AFC</b>', icon=escudosunderland).add_to(m)

logo_urldoncaster = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/46/Doncaster_Rovers_FC.png/150px-Doncaster_Rovers_FC.png'

escudodoncaster = folium.features.CustomIcon(logo_urldoncaster,icon_size=(40,40))

folium.Marker([53.509722, -1.113889], popup='<br><a href= https://en.wikipedia.org/wiki/Doncaster_Rovers_F.C. >https://en.wikipedia.org/wiki/Doncaster_Rovers_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 15231'+'<br><b>Official Web:</b> https://www.doncasterroversfc.co.uk/', tooltip='<b>Keepmoat Stadium - DONCASTER ROVERS</b>', icon=escudodoncaster).add_to(m)

logo_urlgillingham = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/5e/FC_Gillingham_Logo.svg/130px-FC_Gillingham_Logo.svg.png'

escudogillingham = folium.features.CustomIcon(logo_urlgillingham,icon_size=(40,40))

folium.Marker([51.38425, 0.560753], popup='<br><a href= https://en.wikipedia.org/wiki/Gillingham_F.C. >https://en.wikipedia.org/wiki/Gillingham_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 11582'+'<br><b>Official Web:</b> https://www.gillinghamfootballclub.com/', tooltip='<b>Priestfield Stadium - GILLINGHAM FC</b>', icon=escudogillingham).add_to(m)

logo_urlipswich = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/43/Ipswich_Town.svg/170px-Ipswich_Town.svg.png'

escudoipswich = folium.features.CustomIcon(logo_urlipswich,icon_size=(40,40))

folium.Marker([52.055061, 1.144831], popup='<br><a href= https://en.wikipedia.org/wiki/Ipswich_Town_F.C. >https://en.wikipedia.org/wiki/Ipswich_Town_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 30311'+'<br><b>Official Web:</b> https://www.itfc.co.uk/', tooltip='<b>Portman Road - IPSWICH TOWN</b>', icon=escudoipswich).add_to(m)

logo_urlburton = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Burton_Albion_FC_logo.svg/220px-Burton_Albion_FC_logo.svg.png'

escudoburton = folium.features.CustomIcon(logo_urlburton,icon_size=(40,40))

folium.Marker([52.821906, -1.626958], popup='<br><a href= https://en.wikipedia.org/wiki/Burton_Albion_F.C. >https://en.wikipedia.org/wiki/Burton_Albion_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 6912'+'<br><b>Official Web:</b> https://www.burtonalbionfc.co.uk/', tooltip='<b>Pirelli Stadium - BURTON ALBION</b>', icon=escudoburton).add_to(m)

logo_urlblackpool = 'https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Blackpool_FC_logo.svg/180px-Blackpool_FC_logo.svg.png'

escudoblackpool = folium.features.CustomIcon(logo_urlblackpool,icon_size=(40,40))

folium.Marker([53.804722, -3.048056], popup='<br><a href= https://en.wikipedia.org/wiki/Blackpool_F.C. >https://en.wikipedia.org/wiki/Blackpool_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 16616'+'<br><b>Official Web:</b> https://www.blackpoolfc.co.uk/', tooltip='<b>Bloomfield Road - BLACKPOOL</b>', icon=escudoblackpool).add_to(m)

logo_urlrovers = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/47/Bristol_Rovers_F.C._logo.svg/200px-Bristol_Rovers_F.C._logo.svg.png'

escudorovers = folium.features.CustomIcon(logo_urlrovers,icon_size=(40,40))

folium.Marker([51.48622, -2.583134], popup='<br><a href= https://en.wikipedia.org/wiki/Bristol_Rovers_F.C. >https://en.wikipedia.org/wiki/Bristol_Rovers_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 12300'+'<br><b>Official Web:</b> https://www.bristolrovers.co.uk/', tooltip='<b>Memorial Stadium - BRISTOL ROVERS</b>', icon=escudorovers).add_to(m)

logo_urlshrewsbury = 'https://upload.wikimedia.org/wikipedia/en/thumb/0/0a/Shrewsbury_Town_F.C._Badge.png/180px-Shrewsbury_Town_F.C._Badge.png'

escudoshrewsbury = folium.features.CustomIcon(logo_urlshrewsbury,icon_size=(40,40))

folium.Marker([52.68863, -2.74933], popup='<br><a href= https://en.wikipedia.org/wiki/Shrewsbury_Town_F.C. >https://en.wikipedia.org/wiki/Shrewsbury_Town_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 9875'+'<br><b>Official Web:</b> https://www.shrewsburytown.com/', tooltip='<b>New Meadow - SHREWSBURY TOWN</b>', icon=escudoshrewsbury).add_to(m)

logo_urllincoln = 'https://upload.wikimedia.org/wikipedia/en/thumb/0/04/Lincoln_city_%282014%29.png/120px-Lincoln_city_%282014%29.png'

escudolincoln = folium.features.CustomIcon(logo_urllincoln,icon_size=(40,40))

folium.Marker([53.218289, -0.540811], popup='<br><a href= https://en.wikipedia.org/wiki/Lincoln_City_F.C. >https://en.wikipedia.org/wiki/Lincoln_City_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 10120'+'<br><b>Official Web:</b> https://www.weareimps.com/', tooltip='<b>Sincil Bank - LINCOLN CITY</b>', icon=escudolincoln).add_to(m)

logo_urlaccrington = 'https://upload.wikimedia.org/wikipedia/en/thumb/6/63/Accstanley.png/200px-Accstanley.png'

escudoaccrington = folium.features.CustomIcon(logo_urlaccrington,icon_size=(40,40))

folium.Marker([53.765356, -2.370911], popup='<br><a href= https://en.wikipedia.org/wiki/Accrington_Stanley_F.C. >https://en.wikipedia.org/wiki/Accrington_Stanley_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 5450'+'<br><b>Official Web:</b> https://www.accringtonstanley.co.uk/', tooltip='<b>Crown Ground - ACCRINGTON STANLEY</b>', icon=escudoaccrington).add_to(m)

logo_urlrochdale = 'https://upload.wikimedia.org/wikipedia/en/thumb/d/d5/Rochdale_badge.png/185px-Rochdale_badge.png'

escudorochdale = folium.features.CustomIcon(logo_urlrochdale,icon_size=(40,40))

folium.Marker([53.620833, -2.18], popup='<br><a href= https://en.wikipedia.org/wiki/Rochdale_A.F.C. >https://en.wikipedia.org/wiki/Rochdale_A.F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 10249'+'<br><b>Official Web:</b> https://www.rochdaleafc.co.uk/', tooltip='<b>Spotland Stadium - ROCHDALE AFC</b>', icon=escudorochdale).add_to(m)

logo_urlmiltonkeynes = 'https://upload.wikimedia.org/wikipedia/en/thumb/b/bf/MK_Dons.png/200px-MK_Dons.png'

escudomiltonkeynes = folium.features.CustomIcon(logo_urlmiltonkeynes,icon_size=(40,40))

folium.Marker([52.009444, -0.733333], popup='<br><a href= https://en.wikipedia.org/wiki/Milton_Keynes_Dons_F.C. >https://en.wikipedia.org/wiki/Milton_Keynes_Dons_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 30500'+'<br><b>Official Web:</b> https://www.mkdons.com/', tooltip='<b>Stadium MK - MILTON KEYNES DONS</b>', icon=escudomiltonkeynes).add_to(m)

logo_urlwimbledon = 'https://upload.wikimedia.org/wikipedia/en/4/4a/AFC_Wimbledon_2020.png'

escudowimbledon = folium.features.CustomIcon(logo_urlwimbledon,icon_size=(40,40))

folium.Marker([51.405083, -0.281944], popup='<br><a href= https://en.wikipedia.org/wiki/AFC_Wimbledon >https://en.wikipedia.org/wiki/AFC_Wimbledon</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 4850'+'<br><b>Official Web:</b> https://www.afcwimbledon.co.uk/', tooltip='<b>Kingsmeadow - AFC WIMBLEDON</b>', icon=escudowimbledon).add_to(m)

logo_urltranmere = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/30/Tranmere_Rovers_FC_logo.svg/200px-Tranmere_Rovers_FC_logo.svg.png'

escudotranmere = folium.features.CustomIcon(logo_urltranmere,icon_size=(40,40))

folium.Marker([53.373611, -3.0325], popup='<br><a href= https://en.wikipedia.org/wiki/Tranmere_Rovers_F.C. >https://en.wikipedia.org/wiki/Tranmere_Rovers_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 16587'+'<br><b>Official Web:</b> https://www.tranmererovers.co.uk/', tooltip='<b>Prenton Park - TRANMERE ROVERS</b>', icon=escudotranmere).add_to(m)

logo_urlsouthend = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/79/Southend_United.svg/250px-Southend_United.svg.png'

escudosouthend = folium.features.CustomIcon(logo_urlsouthend,icon_size=(40,40))

folium.Marker([51.549017, 0.701558], popup='<br><a href= https://en.wikipedia.org/wiki/Southend_United_F.C. >https://en.wikipedia.org/wiki/Southend_United_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 12392'+'<br><b>Official Web:</b> https://www.southendunited.co.uk/', tooltip='<b>Roots Hall - SOUTHEND UNITED</b>', icon=escudosouthend).add_to(m)

logo_urlbolton = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Bolton_Wanderers_FC_logo.svg/180px-Bolton_Wanderers_FC_logo.svg.png'

escudobolton = folium.features.CustomIcon(logo_urlbolton,icon_size=(40,40))

folium.Marker([53.580556, -2.535556], popup='<br><a href= https://en.wikipedia.org/wiki/Bolton_Wanderers_F.C. >https://en.wikipedia.org/wiki/Bolton_Wanderers_F.C.</a>'+'<br><b>Division:</b> League One'+'<br><b>Capacity:</b> 28723'+'<br><b>Official Web:</b> https://www.bwfc.co.uk/', tooltip='<b>University of Bolton Stadium - BOLTON WANDERERS</b>', icon=escudobolton).add_to(m)

#LeagueTwo

logo_urlswindon = 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a3/Swindon_Town_FC.svg/180px-Swindon_Town_FC.svg.png'

escudoswindon = folium.features.CustomIcon(logo_urlswindon,icon_size=(40,40))

folium.Marker([51.564444, -1.770556], popup='<br><a href= https://en.wikipedia.org/wiki/Swindon_Town_F.C. >https://en.wikipedia.org/wiki/Swindon_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 15728'+'<br><b>Official Web:</b> https://www.swindontownfc.co.uk/', tooltip='<b>County Ground - SWINDON TOWN</b>', icon=escudoswindon).add_to(m)

logo_urlcrewe = 'https://upload.wikimedia.org/wikipedia/en/thumb/9/9d/Crewe_Alexandra.svg/180px-Crewe_Alexandra.svg.png'

escudocrewe = folium.features.CustomIcon(logo_urlcrewe,icon_size=(40,40))

folium.Marker([53.087419, -2.435747], popup='<br><a href= https://en.wikipedia.org/wiki/Crewe_Alexandra_F.C. >https://en.wikipedia.org/wiki/Crewe_Alexandra_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 10153'+'<br><b>Official Web:</b> https://www.crewealex.net/', tooltip='<b>Gresty Road - CREWE ALEXANDRA</b>', icon=escudocrewe).add_to(m)

logo_urlplymouth = 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a8/Plymouth_Argyle_F.C._logo.svg/200px-Plymouth_Argyle_F.C._logo.svg.png'

escudoplymouth = folium.features.CustomIcon(logo_urlplymouth,icon_size=(40,40))

folium.Marker([50.388056, -4.150833], popup='<br><a href= https://en.wikipedia.org/wiki/Plymouth_Argyle_F.C. >https://en.wikipedia.org/wiki/Plymouth_Argyle_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 17904'+'<br><b>Official Web:</b> https://www.pafc.co.uk//', tooltip='<b>Home Park - PLYMOUTH ARGYLE</b>', icon=escudoplymouth).add_to(m)

logo_urlcheltenham = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c3/Cheltenham_Town_F.C._logo.svg/150px-Cheltenham_Town_F.C._logo.svg.png'

escudocheltenham = folium.features.CustomIcon(logo_urlcheltenham,icon_size=(40,40))

folium.Marker([51.906158, -2.060211], popup='<br><a href= https://en.wikipedia.org/wiki/Cheltenham_Town_F.C. >https://en.wikipedia.org/wiki/Cheltenham_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 7066'+'<br><b>Official Web:</b> https://www.ctfc.com/', tooltip='<b>Whaddon Road - CHELTENHAM TOWN</b>', icon=escudocheltenham).add_to(m)

logo_urlexeter = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Exeter_City_FC.svg/200px-Exeter_City_FC.svg.png'

escudoexeter = folium.features.CustomIcon(logo_urlexeter,icon_size=(40,40))

folium.Marker([50.730714, -3.52115], popup='<br><a href= https://en.wikipedia.org/wiki/Exeter_City_F.C. >https://en.wikipedia.org/wiki/Exeter_City_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 8696'+'<br><b>Official Web:</b> https://www.exetercityfc.co.uk/', tooltip='<b>St James Park - EXETER CITY</b>', icon=escudoexeter).add_to(m)

logo_urlcolchester = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Colchester_United_FC_logo.svg/170px-Colchester_United_FC_logo.svg.png'

escudocolchester = folium.features.CustomIcon(logo_urlcolchester,icon_size=(40,40))

folium.Marker([51.923394, 0.897703], popup='<br><a href= https://en.wikipedia.org/wiki/Colchester_United_F.C. >https://en.wikipedia.org/wiki/Colchester_United_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 10105'+'<br><b>Official Web:</b> https://www.cu-fc.com/', tooltip='<b>Colchester Community Stadium - COLCHESTER UNITED</b>', icon=escudocolchester).add_to(m)

logo_urlnorthampton = 'https://upload.wikimedia.org/wikipedia/en/thumb/c/ce/Northampton_Town_FC_logo.png/150px-Northampton_Town_FC_logo.png'

escudonorthampton = folium.features.CustomIcon(logo_urlnorthampton,icon_size=(40,40))

folium.Marker([52.235197, -0.93345], popup='<br><a href= https://en.wikipedia.org/wiki/Northampton_Town_F.C. >https://en.wikipedia.org/wiki/Northampton_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 7798'+'<br><b>Official Web:</b> https://www.ntfc.co.uk/', tooltip='<b>Sixfields Stadium - NORTHAMPTON TOWN</b>', icon=escudonorthampton).add_to(m)

logo_urlportvale = 'https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Port_Vale_logo.svg/200px-Port_Vale_logo.svg.png'

escudoportvale = folium.features.CustomIcon(logo_urlportvale,icon_size=(40,40))

folium.Marker([53.049722, -2.1925], popup='<br><a href= https://en.wikipedia.org/wiki/Port_Vale_F.C. >https://en.wikipedia.org/wiki/Port_Vale_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 19052'+'<br><b>Official Web:</b> https://www.port-vale.co.uk/', tooltip='<b>Vale Park - PORT VALE FC</b>', icon=escudoportvale).add_to(m)

logo_urlbradford = 'https://upload.wikimedia.org/wikipedia/en/thumb/3/32/Bradford_City_AFC.png/130px-Bradford_City_AFC.png'

escudobradford = folium.features.CustomIcon(logo_urlbradford,icon_size=(40,40))

folium.Marker([53.804167, -1.758889], popup='<br><a href= https://en.wikipedia.org/wiki/Bradford_City_A.F.C. >https://en.wikipedia.org/wiki/Bradford_City_A.F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 25136'+'<br><b>Official Web:</b> https://www.bradfordcityfc.co.uk/', tooltip='<b>Valley Parade - BRADFORD CITY</b>', icon=escudobradford).add_to(m)

logo_urlforestgreen = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/85/Forest_Green_Rovers_crest.svg/200px-Forest_Green_Rovers_crest.svg.png'

escudoforestgreen = folium.features.CustomIcon(logo_urlforestgreen,icon_size=(40,40))

folium.Marker([51.698975, -2.237892], popup='<br><a href= https://en.wikipedia.org/wiki/Forest_Green_Rovers_F.C. >https://en.wikipedia.org/wiki/Forest_Green_Rovers_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 5032'+'<br><b>Official Web:</b> https://www.fgr.co.uk/', tooltip='<b>The New Lawn - FOREST GREEN ROVERS</b>', icon=escudoforestgreen).add_to(m)

logo_urlsalford = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f3/Salford_City_FC_Logo.png/150px-Salford_City_FC_Logo.png'

escudosalford = folium.features.CustomIcon(logo_urlsalford,icon_size=(40,40))

folium.Marker([53.513631, -2.276775], popup='<br><a href= https://en.wikipedia.org/wiki/Salford_City_F.C. >https://en.wikipedia.org/wiki/Salford_City_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 5108'+'<br><b>Official Web:</b> https://salfordcityfc.co.uk/', tooltip='<b>Moor Lane - SALFORD CITY</b>', icon=escudosalford).add_to(m)

logo_urlwalsall = 'https://upload.wikimedia.org/wikipedia/en/thumb/e/ef/Walsall_FC.svg/150px-Walsall_FC.svg.png'

escudowalsall = folium.features.CustomIcon(logo_urlwalsall,icon_size=(40,40))

folium.Marker([52.5655, -1.9909], popup='<br><a href= https://en.wikipedia.org/wiki/Walsall_F.C. >https://en.wikipedia.org/wiki/Walsall_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 11300'+'<br><b>Official Web:</b> https://www.saddlers.co.uk/', tooltip='<b>Bescot Stadium - WALSALL FC</b>', icon=escudowalsall).add_to(m)

logo_urlcrawley = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/Crawley_Town_FC_logo.png/200px-Crawley_Town_FC_logo.png'

escudocrawley = folium.features.CustomIcon(logo_urlcrawley,icon_size=(40,40))

folium.Marker([51.099706, -0.194767], popup='<br><a href= https://en.wikipedia.org/wiki/Crawley_Town_F.C. >https://en.wikipedia.org/wiki/Crawley_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 6134'+'<br><b>Official Web:</b> https://www.crawleytownfc.com/', tooltip='<b>Broadfield Stadium - CRAWLEY TOWN</b>', icon=escudocrawley).add_to(m)

logo_urlnewport = 'https://upload.wikimedia.org/wikipedia/en/thumb/b/b0/Newport_County_crest.png/200px-Newport_County_crest.png'

escudonewport = folium.features.CustomIcon(logo_urlnewport,icon_size=(40,40))

folium.Marker([51.588333, -2.987778], popup='<br><a href= https://en.wikipedia.org/wiki/Newport_County_A.F.C. >https://en.wikipedia.org/wiki/Newport_County_A.F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 8700'+'<br><b>Official Web:</b> https://www.newport-county.co.uk/', tooltip='<b>Rodney Parade - NEWPORT COUNTY AFC</b>', icon=escudonewport).add_to(m)

logo_urlgrimsby = 'https://upload.wikimedia.org/wikipedia/en/thumb/d/dc/Grimb_Badge.png/180px-Grimb_Badge.png'

escudogrimsby = folium.features.CustomIcon(logo_urlgrimsby,icon_size=(40,40))

folium.Marker([53.570053, -0.046333], popup='<br><a href= https://en.wikipedia.org/wiki/Grimsby_Town_F.C. >https://en.wikipedia.org/wiki/Grimsby_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 9052'+'<br><b>Official Web:</b> https://www.grimsby-townfc.co.uk/', tooltip='<b>Blundell Park - GRIMSBY TOWN</b>', icon=escudogrimsby).add_to(m)

logo_urlcambridge = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8f/Cambridge_United_FC.svg/160px-Cambridge_United_FC.svg.png'

escudocambridge = folium.features.CustomIcon(logo_urlcambridge,icon_size=(40,40))

folium.Marker([52.2121, 0.15415], popup='<br><a href= https://en.wikipedia.org/wiki/Cambridge_United_F.C. >https://en.wikipedia.org/wiki/Cambridge_United_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 8127'+'<br><b>Official Web:</b> https://www.cambridge-united.co.uk/', tooltip='<b>Abbey Stadium - CAMBRIDGE UNITED</b>', icon=escudocambridge).add_to(m)

logo_urlleyton = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f1/Leyton_Orient_FC.png/150px-Leyton_Orient_FC.png'

escudoleyton = folium.features.CustomIcon(logo_urlleyton,icon_size=(40,40))

folium.Marker([51.56015, -0.012658], popup='<br><a href= https://en.wikipedia.org/wiki/Leyton_Orient_F.C. >https://en.wikipedia.org/wiki/Leyton_Orient_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 9271'+'<br><b>Official Web:</b> https://www.leytonorient.com/', tooltip='<b>Brisbane Road - LEYTON ORIENT</b>', icon=escudoleyton).add_to(m)

logo_urlcarlisle = 'https://upload.wikimedia.org/wikipedia/en/thumb/6/63/Carl_Badge.png/160px-Carl_Badge.png'

escudocarlisle = folium.features.CustomIcon(logo_urlcarlisle,icon_size=(40,40))

folium.Marker([54.89556, -2.91365], popup='<br><a href= https://en.wikipedia.org/wiki/Carlisle_United_F.C. >https://en.wikipedia.org/wiki/Carlisle_United_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 17949'+'<br><b>Official Web:</b> https://www.carlisleunited.co.uk/', tooltip='<b>Brunton Park - CARLISLE UNITED</b>', icon=escudocarlisle).add_to(m)

logo_urloldham = 'https://upload.wikimedia.org/wikipedia/en/thumb/2/21/Oldham_Athletic_new_badge.png/150px-Oldham_Athletic_new_badge.png'

escudooldham = folium.features.CustomIcon(logo_urloldham,icon_size=(40,40))

folium.Marker([53.555278, -2.128611], popup='<br><a href= https://en.wikipedia.org/wiki/Oldham_Athletic_A.F.C. >https://en.wikipedia.org/wiki/Oldham_Athletic_A.F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 13513'+'<br><b>Official Web:</b> https://www.oldhamathletic.co.uk/', tooltip='<b>Boundary Park - OLDHAM ATHLETIC</b>', icon=escudooldham).add_to(m)

logo_urlscunthorpe = 'https://upload.wikimedia.org/wikipedia/en/thumb/9/95/Scunthorpe_United_FC_logo.svg/220px-Scunthorpe_United_FC_logo.svg.png'

escudoscunthorpe = folium.features.CustomIcon(logo_urlscunthorpe,icon_size=(40,40))

folium.Marker([53.5865, -0.695333], popup='<br><a href= https://en.wikipedia.org/wiki/Scunthorpe_United_F.C. >https://en.wikipedia.org/wiki/Scunthorpe_United_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 9088'+'<br><b>Official Web:</b> https://www.scunthorpe-united.co.uk/', tooltip='<b>Glanford Park - SCUNTHORPE UNITED</b>', icon=escudoscunthorpe).add_to(m)

logo_urlmansfield = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Mansfield_Town_FC.svg/150px-Mansfield_Town_FC.svg.png'

escudomansfield = folium.features.CustomIcon(logo_urlmansfield,icon_size=(40,40))

folium.Marker([53.138056, -1.200556], popup='<br><a href= https://en.wikipedia.org/wiki/Mansfield_Town_F.C. >https://en.wikipedia.org/wiki/Mansfield_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 9186'+'<br><b>Official Web:</b> https://www.mansfieldtown.net/', tooltip='<b>Field Mill - MANSFIELD TOWN</b>', icon=escudomansfield).add_to(m)

logo_urlmorecambe = 'https://upload.wikimedia.org/wikipedia/en/thumb/f/f1/Morecambe_FC_Badge.png/175px-Morecambe_FC_Badge.png'

escudomorecambe = folium.features.CustomIcon(logo_urlmorecambe,icon_size=(50,50))

folium.Marker([54.0615, -2.8672], popup='<br><a href= https://en.wikipedia.org/wiki/Morecambe_F.C. >https://en.wikipedia.org/wiki/Morecambe_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 6476'+'<br><b>Official Web:</b> https://www.morecambefc.com/', tooltip='<b>Globe Arena - MORECAMBE</b>', icon=escudomorecambe).add_to(m)

logo_urlmacclesfield = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/40/Macclesfield_Town_FC.svg/180px-Macclesfield_Town_FC.svg.png'

escudomacclesfield = folium.features.CustomIcon(logo_urlmacclesfield,icon_size=(50,50))

folium.Marker([53.242781, -2.127136], popup='<br><a href= https://en.wikipedia.org/wiki/Macclesfield_Town_F.C. >https://en.wikipedia.org/wiki/Macclesfield_Town_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 6355'+'<br><b>Official Web:</b> https://www.mtfc.co.uk/', tooltip='<b>Moss Rose - MACCLESFIELD TOWN</b>', icon=escudomacclesfield).add_to(m)

logo_urlstevenage = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/45/Stevenage_Football_Club.png/150px-Stevenage_Football_Club.png'

escudostevenage = folium.features.CustomIcon(logo_urlstevenage,icon_size=(50,50))

folium.Marker([51.889839, -0.193608], popup='<br><a href= https://en.wikipedia.org/wiki/Stevenage_F.C. >https://en.wikipedia.org/wiki/Stevenage_F.C.</a>'+'<br><b>Division:</b> League Two'+'<br><b>Capacity:</b> 7800'+'<br><b>Official Web:</b> https://www.stevenagefc.com/', tooltip='<b>Broadhall Way - STEVENAGE FC</b>', icon=escudostevenage).add_to(m)

m
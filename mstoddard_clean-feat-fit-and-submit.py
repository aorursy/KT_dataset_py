
#Clean Data V2
def clean_v2(train):
	l1=['American Hairless Terrier','American Leopard Hound','Appenzeller Sennenhunde','Azawakh','Barbet','Basset Fauve de Bretagne','Belgian Laekenois','Biewer Terrier','Bolognese','Bracco Italiano','Braque du Bourbonnais','Broholmer','Catahoula Leopard Dog','Caucasian Ovcharka','Central Asian Shepherd Dog','Czechoslovakian Vlcak','Danish-Swedish Farmdog','Deutscher Wachtelhund','Dogo Argentino','Drentsche Patrijshond','Drever','Dutch Shepherd','Estrela Mountain Dog','Eurasier','French Spaniel','German Longhaired Pointer','German Spitz','Grand Basset Griffon Vendeen','Hamiltonstovare','Hovawart','Jagdterrier','Jindo','Kai Ken','Karelian Bear Dog','Kishu Ken','Kromfohrlander','Lancashire Heeler','Mudi','Nederlandse Kooikerhondje','Norrbottenspets','Perro de Presa Canario','Peruvian Inca Orchid','Portuguese Podengo','Portuguese Pointer','Portuguese Sheepdog','Pumi','Pyrenean Mastiff','Rafeiro do Alentejo','Russian Toy','Russian Tsvetnaya Bolonka','Schapendoes','Shikoku','Slovensky Cuvac','Slovensky Kopov','Sloughi','Small Munsterlander Pointer','Spanish Mastiff','Stabyhoun','Swedish Lapphund','Thai Ridgeback','Tornjak','Tosa','Transylvanian Hound','Treeing Tennessee Brindle','Working Kelpie','Australian Cattle Dog','Australian Shepherd','Bearded Collie','Beauceron','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Border Collie','Bouvier des Flandres','Briard','Canaan Dog','Cardigan Welsh Corgi','Collie','Entlebucher Mountain Dog','Finnish Lapphund','German Shepherd Dog','Icelandic Sheepdog','Miniature American Shepherd','Norwegian Buhund','Old English Sheepdog','Pembroke Welsh Corgi','Polish Lowland Sheepdog','Puli','Pyrenean Shepherd','Shetland Sheepdog','Spanish Water Dog','Swedish Vallhund','Afghan Hound','American English Coonhound','American Foxhound','Basenji','Basset Hound','Beagle','Black and Tan Coonhound','Bloodhound','Bluetick Coonhound','Borzoi','Cirneco de l etna','Dachshund','English Foxhound','Greyhound','Harrier','Ibizan Hound','Irish Wolfhound','Norwegian Elkhound','Otterhound','Petit Basset Griffon Vendeen','Pharaoh Hound','Plott','Portuguese Podengo Pequeno','Redbone Coonhound','Rhodesian Ridgeback','Saluki','Scottish Deerhound','Treeing Walker Coonhound','Whippet','American Eskimo Dog','Bichon Frise','Boston Terrier','Bulldog','Chinese Shar-Pei','Chow Chow','Coton de Tulear','Dalmatian','Finnish Spitz','French Bulldog','Keeshond','Lhasa Apso','Lowchen','Norwegian Lundehund','Poodle','Schipperke','Shiba Inu','Tibetan Spaniel','Tibetan Terrier','Xoloitzcuintli','American Water Spaniel','Boykin Spaniel','Brittany','Chesapeake Bay Retriever','Clumber Spaniel','Cocker Spaniel','Curly-Coated Retriever','English Cocker Spaniel','English Setter','English Springer Spaniel','Field Spaniel','Flat-Coated Retriever','German Shorthaired Pointer','German Wirehaired Pointer','Golden Retriever','Gordon Setter','Irish Red and White Setter','Irish Setter','Irish Water Spaniel','Labrador Retriever','Lagotto Romagnolo','Nova Scotia Duck Tolling Retriever','Pointer','Spinone Italiano','Sussex Spaniel','Vizsla','Weimaraner','Welsh Springer Spaniel','Wirehaired Pointing Griffon','Wirehaired Vizsla','Airedale Terrier','American Staffordshire Terrier','Australian Terrier','Bedlington Terrier','Border Terrier','Bull Terrier','Cairn Terrier','Cesky Terrier','Dandie Dinmont Terrier','Glen of Imaal Terrier','Irish Terrier','Kerry Blue Terrier','Lakeland Terrier','Manchester Terrier','Miniature Bull Terrier','Miniature Schnauzer','Norfolk Terrier','Norwich Terrier','Parson Russell Terrier','Rat Terrier','Russell Terrier','Scottish Terrier','Sealyham Terrier','Skye Terrier','Smooth Fox Terrier','Soft Coated Wheaten Terrier','Staffordshire Bull Terrier','Welsh Terrier','West Highland White Terrier','Wire Fox Terrier','Affenpinscher','Brussels Griffon','Cavalier King Charles Spaniel','Chihuahua','Chinese Crested','English Toy Spaniel','Havanese','Italian Greyhound','Japanese Chin','Maltese','Miniature Pinscher','Papillon','Pekingese','Pomeranian','Pug','Shih Tzu','Silky Terrier','Toy Fox Terrier','Yorkshire Terrier','Akita','Alaskan Malamute','Anatolian Shepherd Dog','Bernese Mountain Dog','Black Russian Terrier','Boerboel','Boxer','Bullmastiff','Cane Corso','Chinook','Doberman Pinscher','Dogue de Bordeaux','German Pinscher','Giant Schnauzer','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Komondor','Kuvasz','Leonberger','Mastiff','Neapolitan Mastiff','Newfoundland','Portuguese Water Dog','Rottweiler','Samoyed','Siberian Husky','Standard Schnauzer','Tibetan Mastiff','St. Bernard']
	l2=['FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working']
	
	l3=pd.DataFrame(l1)
	l3['Groups']=pd.DataFrame(l1)
	l3['Breed11']=pd.DataFrame(l2)
	l3.drop(0,axis=1,inplace=True)
	l3['Breed11']=l3['Breed11'].str.upper()
	
	#2013
	l4=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l5=['1','2','3','5','4','11','6','8','9','7','13','14','10','12','16','17','20','18','15','24','19','23','21','25','26','30','28','22','32','29','27','34','31','33','50','35','37','44','41','39','36','43','40','40','46','45','51','38','48','47','47','60','49','56','57','52','58','59','55','54','62','67','64','65','53','63','80','69','71','73','61','66','72','70','78','68','75','76','74','79','77','83','81','82','86','-9999','94','85','103','99','88','90','93','98','96','84','91','87','108','97','95','-9999','92','89','102','105','111','121','104','109','107','100','117','123','114','106','118','101','110','112','119','125','116','128','126','122','-9999','113','124','135','127','129','115','132','131','139','143','120','130','133','142','134','152','138','141','140','137','148','145','149','150','163','165','153','136','147','-9999','144','158','155','151','154','-9999','159','157','166','146','-9999','167','171','164','160','168','162','-9999','161','169','156','170','174','172','176','175','173','177']
	
	l6=pd.DataFrame(l4)
	l6['Breed11']=pd.DataFrame(l4)
	l6['Classement_2013']=pd.DataFrame(l5)
	l6.drop(0,axis=1,inplace=True)
	l6['Breed11']=l6['Breed11'].str.upper()
	
	#2014
	l7=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l8=['1','2','3','4','5','9','6','7','10','8','12','13','11','14','15','16','18','19','17','22','20','23','21','25','26','27','28','24','32','30','29','34','33','35','48','36','37','40','42','39','38','41','45','44','47','46','49','43','50','51','51','60','52','57','54','53','55','56','59','58','62','68','66','63','61','67','76','75','71','72','69','74','73','70','77','65','78','84','80','79','82','83','85','81','87','31','92','86','103','102','88','90','94','104','95','91','89','93','110','99','98','111','101','97','105','100','114','108','106','109','115','116','118','125','119','122','127','112','120','121','123','129','124','130','128','139','64','126','133','138','132','131','134','135','143','142','144','136','141','148','146','149','145','150','152','137','140','147','155','154','161','163','157','166','151','158','107','160','169','153','159','165','117','168','170','172','156','113','162','173','176','164','167','175','96','177','178','171','174','182','179','180','184','181','183']
	
	l9=pd.DataFrame(l7)
	l9['Breed11']=pd.DataFrame(l7)
	l9['Classement_2014']=pd.DataFrame(l8)
	l9.drop(0,axis=1,inplace=True)
	l9['Breed11']=l9['Breed11'].str.upper()
	
	#2015
	l10=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l11=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184']
	
	l12=pd.DataFrame(l10)
	l12['Breed11']=pd.DataFrame(l10)
	l12['Classement_2015']=pd.DataFrame(l11)
	l12.drop(0,axis=1,inplace=True)
	l12['Breed11']=l12['Breed11'].str.upper()
	
	#join with train
	train=pd.merge(train, l3, on = 'Breed11',how='left')
	train=pd.merge(train, l6, on = 'Breed11',how='left')
	train=pd.merge(train, l9, on = 'Breed11',how='left')
	train=pd.merge(train, l12, on = 'Breed11',how='left')
	
	train['Classement_2013']=train['Classement_2013'].fillna(-9999).astype(float)
	train['Classement_2014']=train['Classement_2014'].fillna(-9999).astype(float)
	train['Classement_2015']=train['Classement_2015'].fillna(-9999).astype(float)
	
	#Type2
	l111=['Great Dane','Mastiff','Neapolitan Mastiff','Bullmastiff','Newfoundland','Dogue de Bordeaux','Cane Corso','Great Pyrenees','Bernese Mountain Dog','Tibetan Mastiff','Black Russian Terrier','Leonberger','Irish Wolfhound','Scottish Deerhound','Brussels Griffon','Cavalier King Charles Spaniel','Chihuahua','Chinese Crested','Dachshund','English Toy Spaniel','Havanese','Italian Greyhound','Japanese Chin','Maltese','Miniature Pinscher','Norfolk Terrier','Norwich Terrier','Papillon','Pekingese','Pomeranian','Pug','Schipperke','Shih Tzu','Silky Terrier','Toy Fox Terrier','Manchester Terrier','Poodle','Yorkshire Terrier','Brittany','Bulldog','Cocker Spaniel','English Springer Spaniel','Pembroke Welsh Corgi','Shetland Sheepdog','Whippet']
	l222=['Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Medium','Medium','Medium','Medium','Medium','Medium','Medium']
	
	l13=pd.DataFrame(l111)
	l13['Breed11']=pd.DataFrame(l111)
	l13['Type2']=pd.DataFrame(l222)
	l13.drop(0,axis=1,inplace=True)
	l13['Breed11']=l13['Breed11'].str.upper()
	train=pd.merge(train, l13, on = 'Breed11',how='left')
	train['Type2']=train['Type2'].fillna('Autres')
	
	####Appt
	l_1=['Bichon Frise','Bulldog','Cavalier King Charles Spaniel','Chinese Crested','French Bulldog','Greyhound','Havanese','Maltese','Pug','Shih Tzu']
	l_2=['Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments']
	
	l13_1=pd.DataFrame(l_1)
	l13_1['Breed11']=pd.DataFrame(l_1)
	l13_1['App']=pd.DataFrame(l_2)
	l13_1.drop(0,axis=1,inplace=True)
	l13_1['Breed11']=l13_1['Breed11'].str.upper()
	train=pd.merge(train, l13_1, on = 'Breed11',how='left')
	train['App']=train['App'].fillna('Autres')
	
	####Family
	l_3=['Beagle','Brussels Griffon','Bulldog','Collie','French Bulldog','Golden Retriever','Irish Setter','Labrador Retriever','Newfoundland','Pug']
	l_4=['Family','Family','Family','Family','Family','Family','Family','Family','Family','Family']
	
	l13_2=pd.DataFrame(l_3)
	l13_2['Breed11']=pd.DataFrame(l_3)
	l13_2['Family']=pd.DataFrame(l_4)
	l13_2.drop(0,axis=1,inplace=True)
	l13_2['Breed11']=l13_2['Breed11'].str.upper()
	train=pd.merge(train, l13_2, on = 'Breed11',how='left')
	train['Family']=train['Family'].fillna('Autres')
	
	####Hairless
	l_5=['American Hairless Terrier','Chinese Crested','Peruvian Inca Orchid','Xoloitzcuintli']
	l_6=['Hairless','Hairless','Hairless','Hairless']
	
	l13_3=pd.DataFrame(l_5)
	l13_3['Breed11']=pd.DataFrame(l_5)
	l13_3['Hairless']=pd.DataFrame(l_6)
	l13_3.drop(0,axis=1,inplace=True)
	l13_3['Breed11']=l13_3['Breed11'].str.upper()
	train=pd.merge(train, l13_3, on = 'Breed11',how='left')
	train['Hairless']=train['Hairless'].fillna('Autres')

	####Hypoa
	l_7=['Afghan Hound','American Hairless Terrier','Bedlington Terrier','Bichon Frise','Chinese Crested','Coton de Tulear','Giant Schnauzer','Irish Water Spaniel','Kerry Blue Terrier','Lagotto Romagnolo','Maltese','Peruvian Inca Orchid','Poodle','Portuguese Water Dog','Soft Coated Wheaten Terrier','Standard Schnauzer','Xoloitzcuintli']
	l_8=['Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic']
	
	l13_4=pd.DataFrame(l_7)
	l13_4['Breed11']=pd.DataFrame(l_7)
	l13_4['Hypoa']=pd.DataFrame(l_8)
	l13_4.drop(0,axis=1,inplace=True)
	l13_4['Breed11']=l13_4['Breed11'].str.upper()
	train=pd.merge(train, l13_4, on = 'Breed11',how='left')
	train['Hypoa']=train['Hypoa'].fillna('Autres')
	
	####Kids
	l_9=['Beagle','Boxer','Bull Terrier','Bulldog','Golden Retriever','Labrador Retriever','Newfoundland','Soft Coated Wheaten Terrier','Weimaraner']
	l_10=['Kids','Kids','Kids','Kids','Kids','Kids','Kids','Kids','Kids']
	
	l13_5=pd.DataFrame(l_9)
	l13_5['Breed11']=pd.DataFrame(l_9)
	l13_5['Kids']=pd.DataFrame(l_10)
	l13_5.drop(0,axis=1,inplace=True)
	l13_5['Breed11']=l13_5['Breed11'].str.upper()
	train=pd.merge(train, l13_5, on = 'Breed11',how='left')
	train['Kids']=train['Kids'].fillna('Autres')
	
	####Smart
	l_11=['Bloodhound','Border Collie','Doberman Pinscher','German Shepherd Dog','Golden Retriever','Labrador Retriever','Papillon','Poodle','Rottweiler','Shetland Sheepdog']
	l_12=['Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart']
	
	l13_6=pd.DataFrame(l_11)
	l13_6['Breed11']=pd.DataFrame(l_11)
	l13_6['Smart']=pd.DataFrame(l_12)
	l13_6.drop(0,axis=1,inplace=True)
	l13_6['Breed11']=l13_6['Breed11'].str.upper()
	train=pd.merge(train, l13_6, on = 'Breed11',how='left')
	train['Smart']=train['Smart'].fillna('Autres')
	
	train=train.merge(right1, left_on='Breed', right_index=True,how='left').merge(right2, left_on='Breed',right_index=True,how='left').merge(right3,left_on='Breed', right_index=True,how='left').merge(right4,left_on='Breed',right_index=True,how='left').merge(right5,left_on='Breed', right_index=True,how='left').merge(right6, left_on='Breed',right_index=True,how='left').merge(right7,left_on='Breed', right_index=True,how='left').merge(right8, left_on='Breed',right_index=True,how='left').merge(right9,left_on='Breed', right_index=True,how='left').merge(right10, left_on='Breed',right_index=True,how='left').merge(right11,left_on='Breed', right_index=True,how='left').merge(right12, left_on='Breed',right_index=True,how='left').merge(right13,left_on='Breed', right_index=True,how='left').merge(right14, left_on='Breed',right_index=True,how='left').merge(right15,left_on='Breed', right_index=True,how='left').merge(right16, left_on='Breed',right_index=True,how='left')
	train=train.rename(columns={'Aggressive_y':'a1','At Vet_y':'a2','Barn_y':'a3','Behavior_y':'a4','Court/Investigation_y':'a5','Enroute_y':'a6','Foster_y':'a7','In Foster_y':'a8',
	            'In Kennel_y':'a9','In Surgery_y':'a10','Medical_y':'a11','Offsite_y':'a12','Partner_y':'a13','Rabies Risk_y':'a14','SCRP_y':'a15','Suffering_y':'a16'})
	
	train=train.merge(left1, left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left2, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left3,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left4,left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left5,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left6, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left7,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left8, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left9,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left10, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left11,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left12, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left13,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left14, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left15,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left16, left_on='AgeuponOutcome_num',right_index=True,how='left')
	
	train=train.merge(a1, left_on='Breed11', right_index=True,how='left').merge(a2, left_on='Breed11',right_index=True,how='left').merge(a3,left_on='Breed11', right_index=True,how='left').merge(a4,left_on='Breed11',right_index=True,how='left').merge(a5,left_on='Breed11', right_index=True,how='left').merge(a6, left_on='Breed11',right_index=True,how='left').merge(a7,left_on='Breed11', right_index=True,how='left').merge(a8, left_on='Breed11',right_index=True,how='left').merge(a9,left_on='Breed11', right_index=True,how='left').merge(a10, left_on='Breed11',right_index=True,how='left').merge(a11,left_on='Breed11', right_index=True,how='left').merge(a12, left_on='Breed11',right_index=True,how='left').merge(a13,left_on='Breed11', right_index=True,how='left').merge(a14, left_on='Breed11',right_index=True,how='left').merge(a15,left_on='Breed11', right_index=True,how='left').merge(a16, left_on='Breed11',right_index=True,how='left')
	
	train.drop(['Aggressive_x','At Vet_x','Barn_x','Behavior_x','Court/Investigation_x','Enroute_x','Foster_x','In Foster_x',
	            'In Kennel_x','In Surgery_x','Medical_x','Offsite_x','Partner_x','Rabies Risk_x','SCRP_x','Suffering_x'],axis=1,inplace=True)
	#Type4
	l15=['Akita','Alaskan Malamute','Anatolian Shepherd Dog','Argentine Dogo','Beauceron','Bernese Mountain Dog','Black Russian Terrier','Bloodhound','Borzoi','Bouvier des Flandres','Briard','Bullmastiff','Cane Corso','Caucasian Ovcharka','Central Asian Shepherd Dog','Curly-Coated Retriever','Doberman Pinsch','Greyhound','Otterhounds','Spinone Italiano','Afghan Hound','Airedale Terrier','American Foxhound','American Staffordshire Terrier','Appenzeller Sennenhunde','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Black and Tan Coonhound','Bluetick Coonhound','Boxer','Bracco Italiano','Catahoula Leopard','Chesapeake Bay Retriever','Chinook','Clumber Spaniel','English Setter','Flat Coat Retriever','German Shorthaired Pointer','German Wirehaired Pointer','Golden Retrievers','Gordon Setters','Irish Red and White Setters','Irish Setters','Labrador Retriever','The Standard Poodle','Redbone Coonhound','Thai Ridgeback','Weimaraner','American English Coonhound','Australian Shepherd','Azawakh','Basset Hound','Bearded Collie','Belgian Laekenois','Belgian Malinois','Bull Terrier','Bulldog','Chinese Sharpei','Chow Chow','Collie','Dalmatian','English Springer Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Grand Basset Griffon Vendéen','Harrier','Ibizan Hounds','Irish Water Spaniels','Keeshond','Norwegian Elkhound','Nova Scotia Duck Tolling Retriever','The Pharaoh Hound','The Plott','The Pointer','Portuguese Podengo','The Saluki','Samoyed','Siberian Husky','Sloughis','Stabyhouns','Sussex Spaniel','Treeing Walker Coonhound','Vizslas','Wirehaired Pointing Griffon','Basenji','Beagle','Cardigan Welsh Corgi','Cocker Spaniel','Dandie Dinmont Terrier','English Cocker Spaniel','Finnish Spitz','German Pinscher','Kai Kens','Kerry Blue Terriers','Miniature Bull Terrier','Norwegian Buhunds','Pembroke Welsh Corgi','Petit Bassett Griffon Vendeen','Polish Lowland Sheepdog','Portuguese Podengo (Medio)','Portuguese Water Dog','Pyrenean Shepherd','Sealyham Terrier','Shetland Sheepdog','Skye Terrier','Soft Coated Wheaten Terrier','Spanish Water Dogs','Staffordshire Bull Terrier','Standard Schnauzer','Tibetan Terrier','Treeing Tennessee Brindle','Welsh Springer Spaniel','West Highland White Terrier','Whippet','Wire Fox Terrier','Affenpinscher','American Eskimo','Miniature American Eskimo','Toy American Eskimo','Australian Terrier','Bichon Frise','Bolognese','Border Terrier','Boston Terrier','Brussels Griffon','Cairn Terrier','Cavalier King Charles Spaniel','Cesky Terrier','Chihuahua','Chinese Crested','Coton de Tulear','Dachshund','English Toy Spaniel','French Bulldog','Havanese','Italian Greyhound','Japanese Chin','Lancashire Heeler','Lhasa Apso','Lowchen','Maltese','Manchester Terrier','Standard Manchester Terrier','Toy Manchester Terrier','Miniature Pinscher','Miniature Schnauzer','Norfolk Terrier','Norwich Terrier','Papillon','Parson Russell Terrier','Pekingese','Pomeranian','Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo (Pequeno)','Pug','Rat Terrier','Schipperke','Scottish Terrier','Shiba Inu','Shih Tzu','Silky Terrier','Smooth Fox Terrier','Tibetan Spaniel','Toy Fox Terrier','Yorkshire Terrier','Xoloitzcuintli']
	
	l16=['Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small']
		
	l17=pd.DataFrame(l15)
	l17['Breed11']=pd.DataFrame(l15)
	l17['Type4']=pd.DataFrame(l16)
	l17.drop(0,axis=1,inplace=True)
	l17['Breed11']=l17['Breed11'].str.upper()
	
	train=pd.merge(train, l17, on = 'Breed11',how='left')
	
	#dangerous dogs
	l18=['Dalmatian','Great Dane','Presa Canario','Doberman Pinsch','Chow Chow','Alaskan Malamute','Wolf-Dog Hybrid','German Shepherd','Rottweiler','Pit Bull']
	l19=['Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog']
	
	l20=pd.DataFrame(l18)
	l20['Breed11']=pd.DataFrame(l18)
	l20['DG']=pd.DataFrame(l19)
	l20.drop(0,axis=1,inplace=True)
	l20['Breed11']=l20['Breed11'].str.upper()
	train=pd.merge(train, l20, on = 'Breed11',how='left')
	train['DG']=train['DG'].fillna('Autres')
	return train
	#'Breed11','Breed2','Color1'
#	train = pd.get_dummies(train, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
#	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])


train_2=clean_v2(train_)
train_2 = pd.get_dummies(train_2, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])

test_2=clean_v2(test_)
test_2 = pd.get_dummies(test_2, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])

#Breed-color-shade with Logistic Regression
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

def featuring(train):
    #train['Breed_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed11']]
    #train['Breed_vec2'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed2']]
    #train['Color_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color']]
    #train['Color1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color1']]
    #train['Color2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color2']]
    #train['Shade1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade1']]
    #train['Shade2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade2']]

    #corpustr_br = train['Breed_vec']
    #corpustr_br2 = train['Breed_vec2']
    #corpustr_col = train['Color_vec']
    #corpustr_col1 = train['Color1_vec']
    #corpustr_col2 = train['Color2_vec']
    #corpustr_sh1 = train['Shade1_vec']
    #corpustr_sh2 = train['Shade2_vec']

    #vectorizertr = CountVectorizer(stop_words='english',
    #                             ngram_range = ( 1 , 1 ),analyzer="word", 
    #                             binary=True , token_pattern=r'\w+' )

    #tfidftr_br=vectorizertr.fit_transform(corpustr_br).todense()
    #tfidftr_br2=vectorizertr.fit_transform(corpustr_br2).todense()
    #tfidftr_col=vectorizertr.fit_transform(corpustr_col).todense()
    #tfidftr_col1=vectorizertr.fit_transform(corpustr_col1).todense()
    #tfidftr_col2=vectorizertr.fit_transform(corpustr_col2).todense()
    #tfidftr_sh1=vectorizertr.fit_transform(corpustr_sh1).todense()
    #tfidftr_sh2=vectorizertr.fit_transform(corpustr_sh2).todense()

    #dt1=pd.DataFrame(tfidftr_br,dtype=float).add_suffix('_br1')
    #dt2=pd.DataFrame(tfidftr_col,dtype=float).add_suffix('_col')
    #dt3=pd.DataFrame(tfidftr_col1,dtype=float).add_suffix('_col1')
    #dt4=pd.DataFrame(tfidftr_col2,dtype=float).add_suffix('_col2')
    #dt5=pd.DataFrame(tfidftr_sh1,dtype=float).add_suffix('_sh1')
    #dt6=pd.DataFrame(tfidftr_sh2,dtype=float).add_suffix('_sh2')
    #dt7=pd.DataFrame(tfidftr_br2,dtype=float).add_suffix('_br2')
    binarize=train#.join(dt1)#.join(dt2)#.join(dt3).join(dt4).join(dt5).join(dt6).join(dt7)
    return binarize

train_2['version']='Train'
test_2['version']='Test'

result = train_2.append(test_2, ignore_index=True)

binarize=featuring(result)

binarize_train=binarize[binarize['version']=='Train']
binarize_test=binarize[binarize['version']=='Test']
print(binarize.shape)

train_select=pd.DataFrame(binarize_train.select_dtypes(include=['float64','int64','int32','float32']))
test_select=pd.DataFrame(binarize_test.select_dtypes(include=['float64','int64','int32','float32']))

print(train_select.shape)

train_select=train_select.fillna(-9999)
test_select=test_select.fillna(-9999)

dtrain = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest = xgb.DMatrix(x_test,y_test,missing = -9999)

param = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst = xgb.train(param, dtrain, num_round)
#ypred2 = bst.predict(dtest)
#print(log_loss(y_test, ypred2))
param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}
num_round = 125
#bst2 = xgb.train(param, dtrain, num_round)
#ypred22 = bst2.predict(dtest)
#print(log_loss(y_test, ypred2))
#print(log_loss(y_test, (ypred2+ypred22)/2))
param = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst3 = xgb.train(param, dtrain, num_round)
#ypred222 = bst3.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))
param = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst4 = xgb.train(param, dtrain, num_round)
#ypred2222 = bst4.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222)/4))
param = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}
#watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 125
#bst5 = xgb.train(param, dtrain, num_round)
#ypred22222 = bst5.predict(dtest)

#print(log_loss(y_test, ypred22))
#print(log_loss(y_test, (ypred2+ypred22)/2))
#print(log_loss(y_test, (ypred2+ypred22+ypred222)/3))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222)/4))
#print(log_loss(y_test, (ypred2+ypred22+ypred222+ypred2222+ypred22222)/5))
dtrain = xgb.DMatrix(train_select,Target,missing = -9999)
dtest = xgb.DMatrix(test_select,missing = -9999)
num_round = 125

param1 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}


param2 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param3, dtrain, num_round)
bst5 = xgb.train(param3, dtrain, num_round)

ypred_submit = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5
ypred_submit=pd.DataFrame(ypred_submit)

submission = pd.DataFrame()
submission["id"] = Id_ts.values
submission["Adoption"]= ypred_submit[2]
submission["Died"]= ypred_submit[4]
submission["Euthanasia"]= ypred_submit[1]
submission["Return_to_owner"]= ypred_submit[0]
submission["Transfer"]= ypred_submit[3]

submission.to_csv("sub.csv",index=False)
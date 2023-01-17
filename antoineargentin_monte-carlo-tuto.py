import copy
import random
def coupLicite(config, trouJoue):
    if (config[trouJoue] > 0):
        return True
    else : 
        return False
    
def coupsLicites(config, joueur):
    decalage = 0
    resultat = []
    if(joueur == 2) : 
        decalage = 4
    for indice in range(decalage, decalage + 4):
        if coupLicite(config, indice) : 
            resultat.append(indice)
    return resultat


# transformation de estCoupFinal(v7) en estAffame() et partieFinie() :

# estAffame() retourne 1 si le joueur est affame, 0 sinon.

def estAffame(config, joueur):
    n = len(coupsLicites(config, joueur))
    if (n == 0) : 
        return 1
    return 0

# partieFinie() retourne 2 si joueurArbre a gagné, 
# 1 si joueurArbre est affamé dans la configuration config ou a perdu,
# 0 sinon.

def partieFinie(config, scores):
    if (scores[joueurArbre-1] >= scoreGagnant):
        return 2
    if (scores[joueurSuivant(joueurArbre)-1] >= scoreGagnant):
        return 1
    if estAffame(config, joueurArbre):
        return 1
    if estAffame(config, joueurSuivant(joueurArbre)):
        return 2
    else : 
        return 0 
    
    
def jouerCoup(configuration, trouJoue, lesScores):
    config = copy.copy(configuration)
    scores = copy.copy(lesScores)
    if coupLicite(config, trouJoue) : 
        if (trouJoue < 4):
            Joueur = 1
        else :
            Joueur = 2
        regleBoucle = 0
        nombreDeGrainesPosees = 0
        if (config[trouJoue]>7):
            regleBoucle = config[trouJoue]//8
        
        nbGraines = config[trouJoue]
        config[trouJoue] = 0
        indice = trouJoue
        for i in range(1,nbGraines+1) :
            indice += 1
            if (nombreDeGrainesPosees % 7 == 0 and nombreDeGrainesPosees > 0): 
                indice += 1
            if (indice > 7) :
                indice -= 8
            config[indice] += 1
            nombreDeGrainesPosees += 1

        
        terrainAdverse = False
        if ((indice > 3 and Joueur == 1) or (indice < 4 and Joueur == 2)) :
            terrainAdverse = True

        while((config[indice] == 2 or config[indice] == 3) and terrainAdverse) :
        
            scores[Joueur-1] += config[indice]
            config[indice] = 0
            if (indice>0) :
                indice -= 1
            else :
                indice = 7
        
            if not ((indice > 3 and Joueur == 1) or (indice < 4 and Joueur == 2)) :
                terrainAdverse = False
        return copy.deepcopy(config),copy.deepcopy(scores)
        
    else : 
        
        return "erreurJouerCoup","erreur"
def meilleurChoix(liste) : 
    n = len(liste)
    ratios = []*n
    print(liste)
    for i in range(n) : 
        if (liste[i][1]==0):
            ratios[i]=0
        else:
            ratios[i] = liste[i][0]/liste[i][1]
    return indiceMax(ratios)
        
def indiceMax(liste):
    n = len(liste)
    max = liste[0]
    indmax = 0
    for i in range(1,n):
        if liste[i]>max : 
            max = liste[i]
            indmax = i
    return i


# choixFilsMeilleurRatio() renvoie l'indice de listeIndicesFils (normalement non vide) au meilleur ratio.
def choixFilsMeilleurRatio(arbre, listeIndicesFils):
    ratioMax = 0.0;
    iMax = 404;
    for i in listeIndicesFils:
        sTot = simTot(arbre[i])
        sVic = simVic(arbre[i])
        if (sTot != 0):
            ratio = sVic/sTot
        else:
            ratio = 0
        if (ratio >= ratioMax):
            ratioMax = ratio
            iMax = i
    return iMax


# entrée= 1 ou 2 , sortie: 2 ou 1
def joueurSuivant(joueur):
    return joueur*-1+3
#Noeud : [configuration, scores, indices fils, indice père, simulations victorieuses, simulations totales, personne devant jouer, partieFinie ? (2 pour joueurArbre a gagné, 1 s'il a perdu, 0 si la partie n'est pas finie), dernier trou joué]
def askConfig(Noeud):
    return copy.copy(Noeud[0])
def askScores(Noeud):
    return copy.copy(Noeud[1])
def indicesFils(Noeud):
    return copy.copy(Noeud[2])
def ajouterIndiceFils(Noeud, iFils):
    Noeud[2].append(iFils)
def indicePere(Noeud):
    return Noeud[3]
def simVic(Noeud):
    return Noeud[4]
def simTot(Noeud):
    return Noeud[5]
def joueurDevantJouer(Noeud):
    return Noeud[6]
def valPartieFinie(Noeud):
    return Noeud[7]
def dernierTrouJoue(Noeud):
    return Noeud[8]
global joueurArbre  #Utilisé pour simulation()
global scoreGagnant  #score à atteindre pour gagner
global configInit
global scoresInit
# global arbre ?
global probaChoixExploration

joueurArbre = 2
scoreGagnant = 10
configInit = [4,4,4,4,4,4,4,4]
scoresInit = [0,0]
probaChoixExploration = 0.75
def creationArbre() : 
    arbre = []
    racine = [copy.deepcopy(configInit),copy.deepcopy(scoresInit),[],-1, 0, 0, 1, 0, -800]
    arbre.append(racine)
    return arbre

#Remplacement de estCoupFinal() par estAffame partieFinie()
def creationNouveauNoeud(arbre, config, scores, joueurDevantJouer, pere,dernierCoupJoue) :
    noeud = [copy.deepcopy(config),copy.deepcopy(scores),[],pere, 0, 0, joueurDevantJouer, partieFinie(config, scores),dernierCoupJoue]
    arbre.append(noeud)
    # Mise à jour des indices des fils du père
    ajouterIndiceFils(arbre[pere],len(arbre)-1)
    return arbre
# Renvoie l'indice d'une feuille.

# A chaque noeud où L'ORDI DOIT JOUER, on choisit le fils avec le meilleur ratio avec la proba probaChoixExploration. 
# Sinon, soit on choisit un fils parmi les autres fils, soit on créé un nouveau fils si c'est possible d'en créer un nouveau.

def selection(arbre, iRacine):
        i = iRacine
 #       print(arbre,i)
        noeud = arbre[i]
        listeIndicesFils = indicesFils(noeud)
        
 #listeindicesfils est une copie ...pb?

        while (listeIndicesFils != []):
            r = random.random()
            joueur = joueurDevantJouer(noeud)
            filsMeilleurRatio = choixFilsMeilleurRatio(arbre,listeIndicesFils)
            if (joueur == joueurArbre and r <= probaChoixExploration):
                i = filsMeilleurRatio
                
            else :
            # On choisit un fils au hasard, ou on créé un autre fils différent des autres fils. On choisit un fils de manière équiprobable:
                liste = copy.copy(listeIndicesFils)
                n = len(liste)
                r = random.random()
                liste.remove(filsMeilleurRatio)
                if (r <= 1/n) :
                    # On créé un autre fils si c'est possible. C'est un choix équiprobable à choisir un autre fils parmi liste. Si liste est vide ça marche.
                    config = askConfig(noeud)
                    listeCoupsLicites = coupsLicites(config,joueur)
                    if (len(listeCoupsLicites) == n):
                        # Pas d'autres fils possible.
                        i = random.choice(listeIndicesFils)
                    else : 
                        # On créé un fils différent des fils existant
                        scores = askScores(noeud)
                        bool = 0
                        for coup in listeCoupsLicites:
                            configSuivante,scoresSuivants = jouerCoup(config, coup, scores)
                            for iFils in listeIndicesFils :
                                if not (compare(configSuivante, askConfig(arbre[iFils]))):
                                    arbre = creationNouveauNoeud(arbre, copy.deepcopy(configSuivante), copy.deepcopy(scoresSuivants), joueurSuivant(joueur), i, coup)
                                    i = len(arbre)-1
                                    bool = 1
                                    break
                            if bool == 1 :
                                break
                else :
                    # On choisit un fils parmi ceux qui n'ont pas le meilleur ratio si l'ordi doit jouer.
                    # On choisit parmi tous les fils si ce n'est pas à l'ordi de jouer
                    if joueur == joueurArbre :
                        i = random.choice(liste)
                    else :
                        i = random.choice(listeIndicesFils)
            noeud = arbre[i]
            listeIndicesFils = indicesFils(noeud)
            
        return arbre, i
random.random()
# Renvoie arbre et -1 si iFeuille correspond à une fin de partie.
# Sinon, expansion() créé un fils 0/0 à la feuille, renvoie arbre et l'indice du fils dans l'arbre.
# On pourra plus tard créer plusieurs fils et en choisir un.

def expansion(arbre, iFeuille):
    noeudFeuille = arbre[iFeuille]
    joueur = joueurDevantJouer(noeudFeuille)
    configFeuille = copy.deepcopy(askConfig(noeudFeuille))
    scoresConfig = copy.deepcopy(askScores(noeudFeuille))

    val = partieFinie(configFeuille,scoresConfig)
    if (val>0):
        return arbre, -1
    
    listeCoupsLicites = coupsLicites(configFeuille,joueur)
    #liste non vide
    coup = random.choice(listeCoupsLicites)
    configSuivante, scoresSuivants = jouerCoup(configFeuille, coup, scoresConfig)
    arbre  = creationNouveauNoeud(arbre, copy.deepcopy(configSuivante), copy.deepcopy(scoresSuivants), joueurSuivant(joueur), iFeuille, coup)
    
    return arbre, len(arbre)-1
# Simulation() simule la partie au hasard depuis la configuration iFils jusqu'à la fin, renvoie 1 pour une victoire du joueurArbre et 0 pour une défaite.
# On n'alterne pas l'arbre: victoire correspond à la victoire du joueurArbre
# Pour l'instant un joueur affamé correspond à une fin de partie et la défaite du joueur affamé...

def simulation(arbre, iFils):
    noeud = copy.deepcopy(arbre[iFils])
    joueur = joueurDevantJouer(noeud)
    scores = askScores(noeud)
    config = askConfig(noeud)
    valPartie = partieFinie(config, scores)
    
    while (valPartie == 0):

        #choix du coup
        ListecoupsLicites = coupsLicites(config, joueur)
        coupJoue = random.choice(ListecoupsLicites) #choix au hasard
        config, scores = jouerCoup(config, coupJoue, copy.copy(scores))

        joueur = joueurSuivant(joueur)
        valPartie = partieFinie(config, scores)
        
    return valPartie-1
arbre = creationArbre()
print(arbre)
arbreDesequilibre = [[[3, 0, 3, 0, 4, 4, 4, 4], [0, 8], [], -1, 0, 0, 1, 0, -800]]
simulation(arbre,0)
simulation(arbreDesequilibre,0)
# Maj des ratios des noeuds situé entre les noeuds iFils et iRacine

def retropropagation(arbre, iFils, iRacine, victoire):
    i= iFils
    while (i >= iRacine) :
        noeud = arbre[i]
        noeud[4]+=victoire
        noeud[5]+=1
        i = noeud[3]
    return arbre
arbre = [[[4, 4, 4, 4, 4, 4, 4, 4], [0, 0], [1,2,3,4], -1, 4, 4, 1, 0, -800], [[5, 5, 4, 0, 0, 6, 6, 6], [0, 0], [], 0, 1, 1, 2, 0, 3], [[5, 1, 6, 5, 5, 5, 0, 5], [0, 0], [], 0, 1, 1, 2, 0, 1], [[1, 6, 6, 6, 5, 4, 4, 0], [0, 0], [], 0, 1, 1, 2, 0, 0], [[1, 6, 5, 5, 0, 5, 5, 5], [0, 0], [5], 0, 1, 1, 2, 0, 0],[["whatever"],["ok"],[],4,0,0]]
print(arbre)
retropropagation(arbre, 5,0,0)
# Renvoie l'indice du fils de arbre[iRacine] au meilleur ratio et l'indice SUR LE PLATEAU de ce coup.
# Pour l'instant la partie se termine quand un adversaire est affamé...
# Si tout va bien cette fonction n'est jamais appelé si arbre[iRacine] n'a pas de fils

def choixCoup(arbre, iRacine):
    listeIndicesFils = indicesFils(arbre[iRacine])
    if listeIndicesFils == []:
        # cas où on est arrivé à la toute fin de partie
        print("erreur (synchronisation des scores ?)")
    else :
        iMax = choixFilsMeilleurRatio(arbre, listeIndicesFils)
    
    #ensuite retrouver l'indice du plateau qu'on veut jouer
    indicePlateau = dernierTrouJoue(arbre[iMax])
    
    return iMax, indicePlateau
print(arbre)
choixCoup(arbre, 0)
# Compare deux configurations, renvoie 1 si elles sont identiques, 0 sinon.

def compare(config1, config2):
    n = len(config1)
    for i in range(0,n):
        if (config1[i]!=config2[i]):
            return False
    return True



# recherche l'indice de l'arbre correspondant à la configuration identique à plateau, parmi les fils de iRacine. 
# Créé ce noeud (feuille) s'il n'existe pas.
# À utiliser après le tour du joueur réel.
# PROBLEME sur les joueurs

def rechercheRacine(plateau, arbre, iRacine):
    noeud = arbre[iRacine]
    listeIndicesFils = indicesFils(noeud)
    for iFils in listeIndicesFils :
        if (compare(plateau, askConfig(arbre[iFils]))):
            return arbre, iFils
    
    # Pas trouvé, il faut créer un nouveau fils
    # On créé des fils parmi les coups licites jusqu'à trouver le bon
    config = askConfig(noeud)
    scores = askScores(noeud)
    listeCoupsLicites = coupsLicites(config, joueurDevantJouer(noeud))
    for coup in listeCoupsLicites:
        configSuivante,scoresSuivants = jouerCoup(config, coup, scores)
        if (compare(plateau, configSuivante)):
            arbre = creationNouveauNoeud(arbre, copy.copy(plateau), copy.copy(scoresSuivants), joueurSuivant(joueurDevantJouer(noeud)), iRacine, coup)
            return arbre, len(arbre)-1
    print("erreurRechercheRacine")
    return "erreurRechercheRacine", "erreur"
# tourOrdi(): maj l'arbre, trouvre le meilleur coup à jouer, renvoie l'arbre et ce coup

def tourOrdi(plateau, arbre, iRacine):
    arbre, iRacine = rechercheRacine(plateau, arbre, iRacine)
    for n in range (0,1000):
        arbre, iFeuille = selection(arbre, iRacine)
        arbre, iFils = expansion(arbre, iFeuille)
        if (iFils != -1):
            victoire = simulation(arbre, iFils)
        else :
            #Fin de partie
            if (iRacine == iFeuille):
                print("égalité iRacine iFeuille")
                break
            Feuille = arbre[iFeuille]
            victoire = partieFinie(askConfig(Feuille), askScores(Feuille))-1
            iFils = iFeuille
        arbre = retropropagation(arbre, iFils, iRacine, victoire)
    iRacine, coup = choixCoup(arbre, iRacine)
    # Le fils choisi devient la nouvelle racine.
    return arbre, iRacine, coup


# tourJoueurReel(): Affiche le plateau, demande d'entrer un coup, renvoie ce coup.

def tourJoueurReel(plateau):
    joueur = joueurSuivant(joueurArbre)
    listeCoupsLicites = coupsLicites(plateau, joueur)
    print("Entrez le numéro de la coupelle à jouer:")
    coup = int(input())
    while coup not in listeCoupLicites:
        print("Incorrect. Entrez le numéro de la coupelle à jouer.")
        coup = int(input())
    return coup

def tourRandom(plateau, joueur):
    listeCoupsLicites = coupsLicites(plateau, joueur)
    return random.choice(listeCoupsLicites)
def reboot():
    global joueurArbre  #Utilisé pour simulation()
    global scoreGagnant  #score à atteindre pour gagner
    global configInit
    global scoresInit
    # global arbre ?
    global probaChoixExploration

    joueurArbre = 2
    scoreGagnant = 10
    configInit = [4,4,4,4,4,4,4,4]
    scoresInit = [0,0]
    probaChoixExploration = 0.5

def jouer():
    joueur = 1
    reboot()
    plateau = copy.copy(configInit)
    scores = copy.copy(scoresInit)
    arbre = creationArbre()
    iRacine = 0;
    valFinie = 0
    print(plateau,scores)
    while valFinie == 0 :
        if joueur == joueurArbre :
            arbre, iRacine, coup = tourOrdi(plateau, arbre, iRacine)
            print("L'ordi joue le coup ", coup, ".", " iRacine = ", iRacine, ".")
        else :
            coup = tourRandom(plateau, joueurSuivant(joueurArbre))
        plateau, scores = jouerCoup(plateau, coup, scores)
        print(plateau,scores)
        valFinie = partieFinie(plateau, scores)
        joueur = joueurSuivant(joueur)
    return (plateau, scores, valFinie)

#jouer()
def testEfficacite():
    nbSimulations=20
    victoires = 0
    for i in range(0,nbSimulations):
        plateau, scores, victoire = jouer()
        victoires+=victoire-1
    return victoires/nbSimulations

v = testEfficacite()
v


# ======================================== #

# VARIABLES

# Edit these to change what pokemon you are dealing with



# --- Your Pokemon (Attacker) --- #

# -- Default: Charizard -- #



# Set the ID to be the pokemon you want to calculate

pokeID = 6 



# Set the level of the pokemon (1 to 100)

level = 50



# IVs are inherant skills, there is one for each stat (0 to 31)

# (Order is: HP, Attack, Defense, Special Attack, Special Defense, Speed)

IV = [31, 31, 31, 31, 31, 31]



# EVs are learned skills, you can have a max of 255 for each stat but no more than 510 in total

EV = [0, 252, 0, 252, 0, 4]



# Pokemon Natures can increase or decrease certain stats

nature = "Adamant"



# ======================================== #



# --- Enemy Pokemon (Defender) --- #

# -- Default: Decidueye -- #



# Set the ID to be the pokemon you want to calculate

oppID = 724 



# Set the level of the pokemon (1 to 100)

oppLevel = 50



# IVs are inherant skills, there is one for each stat (0 to 31)

# (Order is: HP, Attack, Defense, Special Attack, Special Defense, Speed)

oppIV = [31, 31, 31, 31, 31, 31]



# EVs are learned skills, you can have a max of 255 for each stat but no more than 510 in total

oppEV = [0, 0, 252, 0, 252, 4]



# Pokemon Natures can increase or decrease certain stats

oppNature = "Hardy"



# ======================================== #
# Lets Import Libraries and Read in the Data



import numpy as np

import pandas as pd

import math as math

from io import StringIO

from IPython.core.display import HTML



pokemon = pd.read_csv('../input/pokemon.csv')

natures = pd.read_csv('../input/natures.csv')



movesets = pd.read_csv('../input/movesets.csv')

moves = pd.read_csv('../input/moves.csv')

typechart = pd.read_csv('../input/type-chart.csv').fillna('None')
# Lets set up some computed variables for later



pokemonName = pokemon[pokemon['id'] == pokeID]['species'].iloc[0]

pokemonForme = pokemon[pokemon['id'] == pokeID]['forme'].iloc[0]



nMod = [

    1,

    natures[natures['nature'] == nature]['attack'].iloc[0],

    natures[natures['nature'] == nature]['defense'].iloc[0],

    natures[natures['nature'] == nature]['spattack'].iloc[0],

    natures[natures['nature'] == nature]['spdefense'].iloc[0],

    natures[natures['nature'] == nature]['speed'].iloc[0]

]



# Base Stats are Specific to each Pokemon ID

base = [

    pokemon[pokemon['id'] == pokeID]['hp'].iloc[0] ,

    pokemon[pokemon['id'] == pokeID]['attack'].iloc[0],

    pokemon[pokemon['id'] == pokeID]['defense'].iloc[0],

    pokemon[pokemon['id'] == pokeID]['spattack'].iloc[0],

    pokemon[pokemon['id'] == pokeID]['spdefense'].iloc[0],

    pokemon[pokemon['id'] == pokeID]['speed'].iloc[0]

]
# Now we can calculate the pokemon's exact stats:



stats = [

    ((base[0] * 2 + IV[0] + math.floor(EV[0] / 4)) * level / 100 ) + 10 + level,

    (((base[1] * 2 + IV[1] + math.floor(EV[1] / 4)) * level / 100) + 5) * nMod[1],

    (((base[2] * 2 + IV[2] + math.floor(EV[2] / 4)) * level / 100) + 5) * nMod[2],

    (((base[3] * 2 + IV[3] + math.floor(EV[3] / 4)) * level / 100) + 5) * nMod[3],

    (((base[4] * 2 + IV[4] + math.floor(EV[4] / 4)) * level / 100) + 5) * nMod[4],

    (((base[5] * 2 + IV[5] + math.floor(EV[5] / 4)) * level / 100) + 5) * nMod[5]

]



ourType1 = str(pokemon[pokemon['id'] == pokeID]['type1'].iloc[0]).lower()

ourType2 = str(pokemon[pokemon['id'] == pokeID]['type2'].iloc[0]).lower()
# We also have to calculate the stats for our opponent pokemon:



oppPokemonName = pokemon[pokemon['id'] == oppID]['species'].iloc[0]

oppPokemonForme = pokemon[pokemon['id'] == oppID]['forme'].iloc[0]



oppnMod = [

    1,

    natures[natures['nature'] == oppNature]['attack'].iloc[0],

    natures[natures['nature'] == oppNature]['defense'].iloc[0],

    natures[natures['nature'] == oppNature]['spattack'].iloc[0],

    natures[natures['nature'] == oppNature]['spdefense'].iloc[0],

    natures[natures['nature'] == oppNature]['speed'].iloc[0]

]



oppBase = [

    pokemon[pokemon['id'] == oppID]['hp'].iloc[0] ,

    pokemon[pokemon['id'] == oppID]['attack'].iloc[0],

    pokemon[pokemon['id'] == oppID]['defense'].iloc[0],

    pokemon[pokemon['id'] == oppID]['spattack'].iloc[0],

    pokemon[pokemon['id'] == oppID]['spdefense'].iloc[0],

    pokemon[pokemon['id'] == oppID]['speed'].iloc[0]

]



oppStats = [

    ((oppBase[0] * 2 + oppIV[0] + math.floor(oppEV[0] / 4)) * oppLevel  / 100 ) + 10 + oppLevel ,

    (((oppBase[1] * 2 + oppIV[1] + math.floor(oppEV[1] / 4)) * oppLevel  / 100) + 5) * oppnMod[1],

    (((oppBase[2] * 2 + oppIV[2] + math.floor(oppEV[2] / 4)) * oppLevel  / 100) + 5) * oppnMod[2],

    (((oppBase[3] * 2 + oppIV[3] + math.floor(oppEV[3] / 4)) * oppLevel  / 100) + 5) * oppnMod[3],

    (((oppBase[4] * 2 + oppIV[4] + math.floor(oppEV[4] / 4)) * oppLevel  / 100) + 5) * oppnMod[4],

    (((oppBase[5] * 2 + oppIV[5] + math.floor(oppEV[5] / 4)) * oppLevel  / 100) + 5) * oppnMod[5]

]



oppDef = int(oppStats[2])

oppSpD = int(oppStats[4])



oppType1 = str(pokemon[pokemon['id'] == oppID]['type1'].iloc[0]).lower()

oppType2 = str(pokemon[pokemon['id'] == oppID]['type2'].iloc[0]).lower()
# Now lets grab all of the moves for your pokemon



moveset = movesets[(movesets['species'] == pokemonName) & (movesets['forme'] == pokemonForme)].iloc[0]

baseMoves = []

tmMoves = []

otherMoves = []



for move in moveset[3:]:

    if pd.notnull(move):

        move = move.split(' - ',1)

        if (move[0] == 'Start'):

            baseMoves.append(move[1])

        elif ("L" in move[0]):

            if((int(move[0].split("L",1)[1]) <= level) & (move[1] not in baseMoves)):

                baseMoves.append(move[1])

        elif ("TM" in move[0]):

            tmMoves.append(move[1])

        else:

            otherMoves.append(move[1])
# Lets turn those moves into html objects and calculate expected damage of each



bmstr = ""



for b in baseMoves:

    

    moveName = moves[moves['move'] == str(b)]["move"].iloc[0]

    moveType = moves[moves['move'] == str(b)]["type"].iloc[0]

    moveCategory = moves[moves['move'] == str(b)]["category"].iloc[0]

    movePower = moves[moves['move'] == str(b)]["power"].iloc[0]

    moveAccuracy = moves[moves['move'] == str(b)]["accuracy"].iloc[0]

    movePP = moves[moves['move'] == str(b)]["pp"].iloc[0]

    movePriority = moves[moves['move'] == str(b)]["priority"].iloc[0]

    moveCritical = moves[moves['move'] == str(b)]["crit"].iloc[0]

    

    # Set Accuracy

    if moveAccuracy != '—':

        accuracyMod = float(moveAccuracy.strip('%'))/100

    else:

        accuracyMod = 1

        

    # Set Move Power

    if movePower != '—':

        powerMod = int(movePower)

    else:

        powerMod = 0

    

    # Set Same Type Attack Bonus (STAB)

    if( (moveType == pokemon[pokemon['id'] == pokeID]['type1'].iloc[0]) or (moveType == pokemon[pokemon['id'] == pokeID]['type2'].iloc[0])):

        stab = 1.5

    else:

        stab = 1

    

    # Calculate Crit Expected Value

    

    if(moveCritical == 0):

        crit = (1/16) *  0.5 + 1       

    elif(moveCritical == 1):

        crit = (1/8) *  0.5 + 1 

    elif(moveCritical == 2):

        crit = (1/2) *  0.5 + 1 

    else:

        crit = 1.5

    

    # Calculate Type Effect of Attack

    

    typeEffect = typechart[(typechart['defense-type1'] == oppType1) & (typechart['defense-type2'] == oppType2)][str(moveType).lower()].iloc[0]

    

    # Calculate Expected Damage Based on Move Category

    

    if (moveCategory == "Physical"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[1])/oppDef) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    elif (moveCategory == "Special"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[3])/oppSpD) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    else:

        ExpectDmg = 0

    

    bmstr+=(        

            "<tr>"

                "<td>" + str(moveName) + "</td>"

                "<td>" + str(moveType) + "</td>"

                "<td>" + str(moveCategory) + "</td>"

                "<td>" + str(movePower) + "</td>"

                "<td>" + str(moveAccuracy) + "</td>"

                "<td>" + str(movePP) + "</td>"

                "<td>" + str(movePriority) + "</td>"

                "<td>" + str(crit) + "</td>"

                "<td>" + str(stab) + "</td>"

                "<td>" + str(typeEffect) + "</td>"

                "<td>" + str(ExpectDmg) + "</td>"

            "</tr>"

    )

    

tmstr = ""



for t in tmMoves:

    

    moveName = moves[moves['move'] == str(t)]["move"].iloc[0]

    moveType = moves[moves['move'] == str(t)]["type"].iloc[0]

    moveCategory = moves[moves['move'] == str(t)]["category"].iloc[0]

    movePower = moves[moves['move'] == str(t)]["power"].iloc[0]

    moveAccuracy = moves[moves['move'] == str(t)]["accuracy"].iloc[0]

    movePP = moves[moves['move'] == str(t)]["pp"].iloc[0]

    movePriority = moves[moves['move'] == str(t)]["priority"].iloc[0]

    moveCritical = moves[moves['move'] == str(t)]["crit"].iloc[0]

    

        # Set Accuracy

    if moveAccuracy != '—':

        accuracyMod = float(moveAccuracy.strip('%'))/100

    else:

        accuracyMod = 1

        

    # Set Move Power

    if movePower != '—':

        powerMod = int(movePower)

    else:

        powerMod = 0

    

    # Set Same Type Attack Bonus (STAB)

    if( (moveType == pokemon[pokemon['id'] == pokeID]['type1'].iloc[0]) or (moveType == pokemon[pokemon['id'] == pokeID]['type2'].iloc[0])):

        stab = 1.5

    else:

        stab = 1

    

    # Calculate Crit Expected Value

    

    if(moveCritical == 0):

        crit = (1/16) *  0.5 + 1       

    elif(moveCritical == 1):

        crit = (1/8) *  0.5 + 1 

    elif(moveCritical == 2):

        crit = (1/2) *  0.5 + 1 

    else:

        crit = 1.5

    

    # Calculate Type Effect of Attack

    

    typeEffect = typechart[(typechart['defense-type1'] == oppType1) & (typechart['defense-type2'] == oppType2)][str(moveType).lower()].iloc[0]

    

    # Calculate Expected Damage Based on Move Category

    

    if (moveCategory == "Physical"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[1])/oppDef) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    elif (moveCategory == "Special"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[3])/oppSpD) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    else:

        ExpectDmg = 0

    

    tmstr+=(        

            "<tr>"

                "<td>" + str(moveName) + "</td>"

                "<td>" + str(moveType) + "</td>"

                "<td>" + str(moveCategory) + "</td>"

                "<td>" + str(movePower) + "</td>"

                "<td>" + str(moveAccuracy) + "</td>"

                "<td>" + str(movePP) + "</td>"

                "<td>" + str(movePriority) + "</td>"

                "<td>" + str(crit) + "</td>"

                "<td>" + str(stab) + "</td>"

                "<td>" + str(typeEffect) + "</td>"

                "<td>" + str(ExpectDmg) + "</td>"

            "</tr>"

    )



omstr = ""



for o in otherMoves:

    

    moveName = moves[moves['move'] == str(o)]["move"].iloc[0]

    moveType = moves[moves['move'] == str(o)]["type"].iloc[0]

    moveCategory = moves[moves['move'] == str(o)]["category"].iloc[0]

    movePower = moves[moves['move'] == str(o)]["power"].iloc[0]

    moveAccuracy = moves[moves['move'] == str(o)]["accuracy"].iloc[0]

    movePP = moves[moves['move'] == str(o)]["pp"].iloc[0]

    movePriority = moves[moves['move'] == str(o)]["priority"].iloc[0]

    moveCritical = moves[moves['move'] == str(o)]["crit"].iloc[0]

    

        # Set Accuracy

    if moveAccuracy != '—':

        accuracyMod = float(moveAccuracy.strip('%'))/100

    else:

        accuracyMod = 1

        

    # Set Move Power

    if movePower != '—':

        powerMod = int(movePower)

    else:

        powerMod = 0

    

    # Set Same Type Attack Bonus (STAB)

    if( (moveType == pokemon[pokemon['id'] == pokeID]['type1'].iloc[0]) or (moveType == pokemon[pokemon['id'] == pokeID]['type2'].iloc[0])):

        stab = 1.5

    else:

        stab = 1

    

    # Calculate Crit Expected Value

    

    if(moveCritical == 0):

        crit = (1/16) *  0.5 + 1       

    elif(moveCritical == 1):

        crit = (1/8) *  0.5 + 1 

    elif(moveCritical == 2):

        crit = (1/2) *  0.5 + 1 

    else:

        crit = 1.5

    

    # Calculate Type Effect of Attack

    

    typeEffect = typechart[(typechart['defense-type1'] == oppType1) & (typechart['defense-type2'] == oppType2)][str(moveType).lower()].iloc[0]

    

    # Calculate Expected Damage Based on Move Category

    

    if (moveCategory == "Physical"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[1])/oppDef) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    elif (moveCategory == "Special"):

        ExpectDmg = int(((((2 * level) + 10) / 250) * (float(stats[3])/oppSpD) * powerMod + 2) * stab * crit * typeEffect * 0.925 * accuracyMod)

        

    else:

        ExpectDmg = 0

    

    omstr+=(        

            "<tr>"

                "<td>" + str(moveName) + "</td>"

                "<td>" + str(moveType) + "</td>"

                "<td>" + str(moveCategory) + "</td>"

                "<td>" + str(movePower) + "</td>"

                "<td>" + str(moveAccuracy) + "</td>"

                "<td>" + str(movePP) + "</td>"

                "<td>" + str(movePriority) + "</td>"

                "<td>" + str(crit) + "</td>"

                "<td>" + str(stab) + "</td>"

                "<td>" + str(typeEffect) + "</td>"

                "<td>" + str(ExpectDmg) + "</td>"

            "</tr>"

    )
from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# Now lets display information about your Pokemon!



def displayPokemon():

    raw_html = (

        "<h1>" + pokemonName + "</h1>"

        "<p>(Level: " + str(level) + ", " + nature + ", " + ourType1 + "/" + ourType2 + ")</p>"

        "<br>"

        "<table>"

            "<tr>"

                "<th></th>"

                "<th>HP</th>"

                "<th>Attack</th>"

                "<th>Defense</th>"

                "<th>Special Attack</th>"

                "<th>Special Defense</th>"

                "<th>Speed</th>"

            "</tr>"

            "<tr>"

                "<td>IVs:</td>"

                "<td>" + str(IV[0]) + "</td>"

                "<td>" + str(IV[1]) + "</td>"

                "<td>" + str(IV[2]) + "</td>"

                "<td>" + str(IV[3]) + "</td>"

                "<td>" + str(IV[4]) + "</td>"

                "<td>" + str(IV[5]) + "</td>"

            "</tr>"

            "<tr>"

                "<td>EVs:</td>"

                "<td>" + str(EV[0]) + "</td>"

                "<td>" + str(EV[1]) + "</td>"

                "<td>" + str(EV[2]) + "</td>"

                "<td>" + str(EV[3]) + "</td>"

                "<td>" + str(EV[4]) + "</td>"

                "<td>" + str(EV[5]) + "</td>"

            "</tr>"

            "<tr>"

                "<td>Base Stats:</td>"

                "<td>" + str(int(base[0])) + "</td>"

                "<td>" + str(int(base[1])) + "</td>"

                "<td>" + str(int(base[2])) + "</td>"

                "<td>" + str(int(base[3])) + "</td>"

                "<td>" + str(int(base[4])) + "</td>"

                "<td>" + str(int(base[5])) + "</td>"

            "</tr>"

            "<tr>"

                "<td>Actual Stats:</td>"

                "<td>" + str(int(stats[0])) + "</td>"

                "<td>" + str(int(stats[1])) + "</td>"

                "<td>" + str(int(stats[2])) + "</td>"

                "<td>" + str(int(stats[3])) + "</td>"

                "<td>" + str(int(stats[4])) + "</td>"

                "<td>" + str(int(stats[5])) + "</td>"

            "</tr>"

        "</table>"

        

        "<h3> Against – " + oppPokemonName + "</h3>"

        "<p>(Level: " + str(oppLevel) + ", " + oppNature + ", " + oppType1 + "/" + oppType2 + ")</p>"

        "<br>"

        "<table>"

            "<tr>"

                "<th></th>"

                "<th>HP</th>"

                "<th>Attack</th>"

                "<th>Defense</th>"

                "<th>Special Attack</th>"

                "<th>Special Defense</th>"

                "<th>Speed</th>"

            "</tr>"

            "<tr>"

                "<td>IVs:</td>"

                "<td>" + str(oppIV[0]) + "</td>"

                "<td>" + str(oppIV[1]) + "</td>"

                "<td>" + str(oppIV[2]) + "</td>"

                "<td>" + str(oppIV[3]) + "</td>"

                "<td>" + str(oppIV[4]) + "</td>"

                "<td>" + str(oppIV[5]) + "</td>"

            "</tr>"

            "<tr>"

                "<td>EVs:</td>"

                "<td>" + str(oppEV[0]) + "</td>"

                "<td>" + str(oppEV[1]) + "</td>"

                "<td>" + str(oppEV[2]) + "</td>"

                "<td>" + str(oppEV[3]) + "</td>"

                "<td>" + str(oppEV[4]) + "</td>"

                "<td>" + str(oppEV[5]) + "</td>"

            "</tr>"

            "<tr>"

                "<td>Base Stats:</td>"

                "<td>" + str(int(oppBase[0])) + "</td>"

                "<td>" + str(int(oppBase[1])) + "</td>"

                "<td>" + str(int(oppBase[2])) + "</td>"

                "<td>" + str(int(oppBase[3])) + "</td>"

                "<td>" + str(int(oppBase[4])) + "</td>"

                "<td>" + str(int(oppBase[5])) + "</td>"

            "</tr>"

            "<tr>"

                "<td>Actual Stats:</td>"

                "<td>" + str(int(oppStats[0])) + "</td>"

                "<td>" + str(int(oppStats[1])) + "</td>"

                "<td>" + str(int(oppStats[2])) + "</td>"

                "<td>" + str(int(oppStats[3])) + "</td>"

                "<td>" + str(int(oppStats[4])) + "</td>"

                "<td>" + str(int(oppStats[5])) + "</td>"

            "</tr>"

        "</table>"

        

        "<h3>Base Moves:</h3>"

        "<table>"

            "<tr>"

                "<th>Move</th>"

                "<th>Type</th>"

                "<th>Category</th>"

                "<th>Power</th>"

                "<th>Accuracy</th>"

                "<th>PP</th>"

                "<th>Priority</th>"

                "<th>Critical</th>"

                "<th>STAB</th>"

                "<th>Type Effect</th>"

                "<th>Expected Damage</th>"

            "</tr>" 

        "" + bmstr + ""

        "</table>"

        

        "<h3>TM Moves:</h3>"

        "<table>"

            "<tr>"

                "<th>Move</th>"

                "<th>Type</th>"

                "<th>Category</th>"

                "<th>Power</th>"

                "<th>Accuracy</th>"

                "<th>PP</th>"

                "<th>Priority</th>"

                "<th>Critical</th>"

                "<th>STAB</th>"

                "<th>Type Effect</th>"

                "<th>Expected Damage</th>"

            "</tr>" 

        "" + tmstr + ""

        "</table>"

        

        "<h3>Other Moves:</h3>"

        "<table>"

            "<tr>"

                "<th>Move</th>"

                "<th>Type</th>"

                "<th>Category</th>"

                "<th>Power</th>"

                "<th>Accuracy</th>"

                "<th>PP</th>"

                "<th>Priority</th>"

                "<th>Critical</th>"

                "<th>STAB</th>"

                "<th>Type Effect</th>"

                "<th>Expected Damage</th>"

            "</tr>" 

        "" + omstr + ""

        "</table>"

     )

    return HTML(raw_html)



displayPokemon()
# ======================================== #

# VARIABLES

# Edit these to change your team

# Select Pokemon using their ID



poke1 = 724 # Decidueye

poke2 = 758 # Salazzle

poke3 = 745 # Lycanroc (Midday)

poke4 = 733 # Toucannon

poke5 = 823 # Ninetales (Alola)

poke6 = 197 # Umbreon



# ======================================== #
# Lets Import Libraries and Read in the Data



import numpy as np

import pandas as pd

import math as math

from io import StringIO

from IPython.core.display import HTML



pokemon = pd.read_csv('../input/pokemon.csv').fillna('none')

natures = pd.read_csv('../input/natures.csv')

typechart = pd.read_csv('../input/type-chart.csv').fillna('none')
# Lets set up some computed variables for later



poke1forme = pokemon[pokemon['id'] == poke1]['forme'].iloc[0]

poke1type1 = str(pokemon[pokemon['id'] == poke1]['type1'].iloc[0]).lower()

poke1type2 = str(pokemon[pokemon['id'] == poke1]['type2'].iloc[0]).lower()



poke2forme = pokemon[pokemon['id'] == poke2]['forme'].iloc[0]

poke2type1 = str(pokemon[pokemon['id'] == poke2]['type1'].iloc[0]).lower()

poke2type2 = str(pokemon[pokemon['id'] == poke2]['type2'].iloc[0]).lower()



poke3forme = pokemon[pokemon['id'] == poke3]['forme'].iloc[0]

poke3type1 = str(pokemon[pokemon['id'] == poke3]['type1'].iloc[0]).lower()

poke3type2 = str(pokemon[pokemon['id'] == poke3]['type2'].iloc[0]).lower()



poke4forme = pokemon[pokemon['id'] == poke4]['forme'].iloc[0]

poke4type1 = str(pokemon[pokemon['id'] == poke4]['type1'].iloc[0]).lower()

poke4type2 = str(pokemon[pokemon['id'] == poke4]['type2'].iloc[0]).lower()



poke5forme = pokemon[pokemon['id'] == poke5]['forme'].iloc[0]

poke5type1 = str(pokemon[pokemon['id'] == poke5]['type1'].iloc[0]).lower()

poke5type2 = str(pokemon[pokemon['id'] == poke5]['type2'].iloc[0]).lower()



poke6forme = pokemon[pokemon['id'] == poke6]['forme'].iloc[0]

poke6type1 = str(pokemon[pokemon['id'] == poke6]['type1'].iloc[0]).lower()

poke6type2 = str(pokemon[pokemon['id'] == poke6]['type2'].iloc[0]).lower()
# Now lets check the defense of your team vs all attacks in the game



def typeEffect(attack, def1, def2):

    

    typeEffect = typechart[(typechart['defense-type1'] == def1) & (typechart['defense-type2'] == def2)][attack].iloc[0]

    

    return typeEffect



normalDef = [typeEffect('normal',poke1type1,poke1type2),typeEffect('normal',poke2type1,poke2type2),typeEffect('normal',poke3type1,poke3type2),typeEffect('normal',poke4type1,poke4type2),typeEffect('normal',poke5type1,poke5type2),typeEffect('normal',poke6type1,poke6type2)]

fireDef = [typeEffect('fire',poke1type1,poke1type2),typeEffect('fire',poke2type1,poke2type2),typeEffect('fire',poke3type1,poke3type2),typeEffect('fire',poke4type1,poke4type2),typeEffect('fire',poke5type1,poke5type2),typeEffect('fire',poke6type1,poke6type2)]

waterDef = [typeEffect('water',poke1type1,poke1type2),typeEffect('water',poke2type1,poke2type2),typeEffect('water',poke3type1,poke3type2),typeEffect('water',poke4type1,poke4type2),typeEffect('water',poke5type1,poke5type2),typeEffect('water',poke6type1,poke6type2)]

electricDef = [typeEffect('electric',poke1type1,poke1type2),typeEffect('electric',poke2type1,poke2type2),typeEffect('electric',poke3type1,poke3type2),typeEffect('electric',poke4type1,poke4type2),typeEffect('electric',poke5type1,poke5type2),typeEffect('electric',poke6type1,poke6type2)]

grassDef = [typeEffect('grass',poke1type1,poke1type2),typeEffect('grass',poke2type1,poke2type2),typeEffect('grass',poke3type1,poke3type2),typeEffect('grass',poke4type1,poke4type2),typeEffect('grass',poke5type1,poke5type2),typeEffect('grass',poke6type1,poke6type2)]

iceDef = [typeEffect('ice',poke1type1,poke1type2),typeEffect('ice',poke2type1,poke2type2),typeEffect('ice',poke3type1,poke3type2),typeEffect('ice',poke4type1,poke4type2),typeEffect('ice',poke5type1,poke5type2),typeEffect('ice',poke6type1,poke6type2)]

fightingDef = [typeEffect('fighting',poke1type1,poke1type2),typeEffect('fighting',poke2type1,poke2type2),typeEffect('fighting',poke3type1,poke3type2),typeEffect('fighting',poke4type1,poke4type2),typeEffect('fighting',poke5type1,poke5type2),typeEffect('fighting',poke6type1,poke6type2)]

poisonDef = [typeEffect('poison',poke1type1,poke1type2),typeEffect('poison',poke2type1,poke2type2),typeEffect('poison',poke3type1,poke3type2),typeEffect('poison',poke4type1,poke4type2),typeEffect('poison',poke5type1,poke5type2),typeEffect('poison',poke6type1,poke6type2)]

groundDef = [typeEffect('ground',poke1type1,poke1type2),typeEffect('ground',poke2type1,poke2type2),typeEffect('ground',poke3type1,poke3type2),typeEffect('ground',poke4type1,poke4type2),typeEffect('ground',poke5type1,poke5type2),typeEffect('ground',poke6type1,poke6type2)]

flyingDef = [typeEffect('flying',poke1type1,poke1type2),typeEffect('flying',poke2type1,poke2type2),typeEffect('flying',poke3type1,poke3type2),typeEffect('flying',poke4type1,poke4type2),typeEffect('flying',poke5type1,poke5type2),typeEffect('flying',poke6type1,poke6type2)]

psychicDef = [typeEffect('psychic',poke1type1,poke1type2),typeEffect('psychic',poke2type1,poke2type2),typeEffect('psychic',poke3type1,poke3type2),typeEffect('psychic',poke4type1,poke4type2),typeEffect('psychic',poke5type1,poke5type2),typeEffect('psychic',poke6type1,poke6type2)]

bugDef = [typeEffect('bug',poke1type1,poke1type2),typeEffect('bug',poke2type1,poke2type2),typeEffect('bug',poke3type1,poke3type2),typeEffect('bug',poke4type1,poke4type2),typeEffect('bug',poke5type1,poke5type2),typeEffect('bug',poke6type1,poke6type2)]

rockDef = [typeEffect('rock',poke1type1,poke1type2),typeEffect('rock',poke2type1,poke2type2),typeEffect('rock',poke3type1,poke3type2),typeEffect('rock',poke4type1,poke4type2),typeEffect('rock',poke5type1,poke5type2),typeEffect('rock',poke6type1,poke6type2)]

ghostDef = [typeEffect('ghost',poke1type1,poke1type2),typeEffect('ghost',poke2type1,poke2type2),typeEffect('ghost',poke3type1,poke3type2),typeEffect('ghost',poke4type1,poke4type2),typeEffect('ghost',poke5type1,poke5type2),typeEffect('ghost',poke6type1,poke6type2)]

dragonDef = [typeEffect('dragon',poke1type1,poke1type2),typeEffect('dragon',poke2type1,poke2type2),typeEffect('dragon',poke3type1,poke3type2),typeEffect('dragon',poke4type1,poke4type2),typeEffect('dragon',poke5type1,poke5type2),typeEffect('dragon',poke6type1,poke6type2)]

darkDef = [typeEffect('dark',poke1type1,poke1type2),typeEffect('dark',poke2type1,poke2type2),typeEffect('dark',poke3type1,poke3type2),typeEffect('dark',poke4type1,poke4type2),typeEffect('dark',poke5type1,poke5type2),typeEffect('dark',poke6type1,poke6type2)]

steelDef = [typeEffect('steel',poke1type1,poke1type2),typeEffect('steel',poke2type1,poke2type2),typeEffect('steel',poke3type1,poke3type2),typeEffect('steel',poke4type1,poke4type2),typeEffect('steel',poke5type1,poke5type2),typeEffect('steel',poke6type1,poke6type2)]

fairyDef = [typeEffect('fairy',poke1type1,poke1type2),typeEffect('fairy',poke2type1,poke2type2),typeEffect('fairy',poke3type1,poke3type2),typeEffect('fairy',poke4type1,poke4type2),typeEffect('fairy',poke5type1,poke5type2),typeEffect('fairy',poke6type1,poke6type2)]
# TO DO: Calculate Attack Bonuses
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

        "<h3>Your Team</h3>"

        "<p>" + poke1forme + " (" + poke1type1 + "/" + poke1type2 + ")<br>"

        "" + poke2forme + " (" + poke2type1 + "/" + poke2type2 + ")<br>"

        "" + poke3forme + " (" + poke3type1 + "/" + poke3type2 + ")<br>"

        "" + poke4forme + " (" + poke4type1 + "/" + poke4type2 + ")<br>"

        "" + poke5forme + " (" + poke5type1 + "/" + poke5type2 + ")<br>"

        "" + poke6forme + " (" + poke6type1 + "/" + poke6type2 + ")</p>"

        "<h3>Defensive Typing</h3>"

        "<table>"

            "<tr>"

                "<th>Attack Type</th>"

                "<th>"+ poke1forme +"</th>"

                "<th>"+ poke2forme +"</th>"

                "<th>"+ poke3forme +"</th>"

                "<th>"+ poke4forme +"</th>"

                "<th>"+ poke5forme +"</th>"

                "<th>"+ poke6forme +"</th>"

                "<th>Mean</th>"

            "</tr>"

            "<tr>"

                "<td>normal</td>"

                "<td>" + str(normalDef[0]) + "</td>"

                "<td>" + str(normalDef[1]) + "</td>"

                "<td>" + str(normalDef[2]) + "</td>"

                "<td>" + str(normalDef[3]) + "</td>"

                "<td>" + str(normalDef[4]) + "</td>"

                "<td>" + str(normalDef[5]) + "</td>"

                "<td>" + str(round(sum(normalDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>fire</td>"

                "<td>" + str(fireDef[0]) + "</td>"

                "<td>" + str(fireDef[1]) + "</td>"

                "<td>" + str(fireDef[2]) + "</td>"

                "<td>" + str(fireDef[3]) + "</td>"

                "<td>" + str(fireDef[4]) + "</td>"

                "<td>" + str(fireDef[5]) + "</td>"

                "<td>" + str(round(sum(fireDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>water</td>"

                "<td>" + str(waterDef[0]) + "</td>"

                "<td>" + str(waterDef[1]) + "</td>"

                "<td>" + str(waterDef[2]) + "</td>"

                "<td>" + str(waterDef[3]) + "</td>"

                "<td>" + str(waterDef[4]) + "</td>"

                "<td>" + str(waterDef[5]) + "</td>"

                "<td>" + str(round(sum(waterDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>electric</td>"

                "<td>" + str(electricDef[0]) + "</td>"

                "<td>" + str(electricDef[1]) + "</td>"

                "<td>" + str(electricDef[2]) + "</td>"

                "<td>" + str(electricDef[3]) + "</td>"

                "<td>" + str(electricDef[4]) + "</td>"

                "<td>" + str(electricDef[5]) + "</td>"

                "<td>" + str(round(sum(electricDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>grass</td>"

                "<td>" + str(grassDef[0]) + "</td>"

                "<td>" + str(grassDef[1]) + "</td>"

                "<td>" + str(grassDef[2]) + "</td>"

                "<td>" + str(grassDef[3]) + "</td>"

                "<td>" + str(grassDef[4]) + "</td>"

                "<td>" + str(grassDef[5]) + "</td>"

                "<td>" + str(round(sum(grassDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>ice</td>"

                "<td>" + str(iceDef[0]) + "</td>"

                "<td>" + str(iceDef[1]) + "</td>"

                "<td>" + str(iceDef[2]) + "</td>"

                "<td>" + str(iceDef[3]) + "</td>"

                "<td>" + str(iceDef[4]) + "</td>"

                "<td>" + str(iceDef[5]) + "</td>"

                "<td>" + str(round(sum(iceDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>fighting</td>"

                "<td>" + str(fightingDef[0]) + "</td>"

                "<td>" + str(fightingDef[1]) + "</td>"

                "<td>" + str(fightingDef[2]) + "</td>"

                "<td>" + str(fightingDef[3]) + "</td>"

                "<td>" + str(fightingDef[4]) + "</td>"

                "<td>" + str(fightingDef[5]) + "</td>"

                "<td>" + str(round(sum(fightingDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>poison</td>"

                "<td>" + str(poisonDef[0]) + "</td>"

                "<td>" + str(poisonDef[1]) + "</td>"

                "<td>" + str(poisonDef[2]) + "</td>"

                "<td>" + str(poisonDef[3]) + "</td>"

                "<td>" + str(poisonDef[4]) + "</td>"

                "<td>" + str(poisonDef[5]) + "</td>"

                "<td>" + str(round(sum(poisonDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>ground</td>"

                "<td>" + str(groundDef[0]) + "</td>"

                "<td>" + str(groundDef[1]) + "</td>"

                "<td>" + str(groundDef[2]) + "</td>"

                "<td>" + str(groundDef[3]) + "</td>"

                "<td>" + str(groundDef[4]) + "</td>"

                "<td>" + str(groundDef[5]) + "</td>"

                "<td>" + str(round(sum(groundDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>flying</td>"

                "<td>" + str(flyingDef[0]) + "</td>"

                "<td>" + str(flyingDef[1]) + "</td>"

                "<td>" + str(flyingDef[2]) + "</td>"

                "<td>" + str(flyingDef[3]) + "</td>"

                "<td>" + str(flyingDef[4]) + "</td>"

                "<td>" + str(flyingDef[5]) + "</td>"

                "<td>" + str(round(sum(flyingDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>psychic</td>"

                "<td>" + str(psychicDef[0]) + "</td>"

                "<td>" + str(psychicDef[1]) + "</td>"

                "<td>" + str(psychicDef[2]) + "</td>"

                "<td>" + str(psychicDef[3]) + "</td>"

                "<td>" + str(psychicDef[4]) + "</td>"

                "<td>" + str(psychicDef[5]) + "</td>"

                "<td>" + str(round(sum(psychicDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>bug</td>"

                "<td>" + str(bugDef[0]) + "</td>"

                "<td>" + str(bugDef[1]) + "</td>"

                "<td>" + str(bugDef[2]) + "</td>"

                "<td>" + str(bugDef[3]) + "</td>"

                "<td>" + str(bugDef[4]) + "</td>"

                "<td>" + str(bugDef[5]) + "</td>"

                "<td>" + str(round(sum(bugDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>rock</td>"

                "<td>" + str(rockDef[0]) + "</td>"

                "<td>" + str(rockDef[1]) + "</td>"

                "<td>" + str(rockDef[2]) + "</td>"

                "<td>" + str(rockDef[3]) + "</td>"

                "<td>" + str(rockDef[4]) + "</td>"

                "<td>" + str(rockDef[5]) + "</td>"

                "<td>" + str(round(sum(rockDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>ghost</td>"

                "<td>" + str(ghostDef[0]) + "</td>"

                "<td>" + str(ghostDef[1]) + "</td>"

                "<td>" + str(ghostDef[2]) + "</td>"

                "<td>" + str(ghostDef[3]) + "</td>"

                "<td>" + str(ghostDef[4]) + "</td>"

                "<td>" + str(ghostDef[5]) + "</td>"

                "<td>" + str(round(sum(ghostDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>dragon</td>"

                "<td>" + str(dragonDef[0]) + "</td>"

                "<td>" + str(dragonDef[1]) + "</td>"

                "<td>" + str(dragonDef[2]) + "</td>"

                "<td>" + str(dragonDef[3]) + "</td>"

                "<td>" + str(dragonDef[4]) + "</td>"

                "<td>" + str(dragonDef[5]) + "</td>"

                "<td>" + str(round(sum(dragonDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>dark</td>"

                "<td>" + str(darkDef[0]) + "</td>"

                "<td>" + str(darkDef[1]) + "</td>"

                "<td>" + str(darkDef[2]) + "</td>"

                "<td>" + str(darkDef[3]) + "</td>"

                "<td>" + str(darkDef[4]) + "</td>"

                "<td>" + str(darkDef[5]) + "</td>"

                "<td>" + str(round(sum(darkDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>steel</td>"

                "<td>" + str(steelDef[0]) + "</td>"

                "<td>" + str(steelDef[1]) + "</td>"

                "<td>" + str(steelDef[2]) + "</td>"

                "<td>" + str(steelDef[3]) + "</td>"

                "<td>" + str(steelDef[4]) + "</td>"

                "<td>" + str(steelDef[5]) + "</td>"

                "<td>" + str(round(sum(steelDef)/6,2)) + "</td>"

            "</tr>"

            "<tr>"

                "<td>fairy</td>"

                "<td>" + str(fairyDef[0]) + "</td>"

                "<td>" + str(fairyDef[1]) + "</td>"

                "<td>" + str(fairyDef[2]) + "</td>"

                "<td>" + str(fairyDef[3]) + "</td>"

                "<td>" + str(fairyDef[4]) + "</td>"

                "<td>" + str(fairyDef[5]) + "</td>"

                "<td>" + str(round(sum(fairyDef)/6,2)) + "</td>"

            "</tr>"

        "</table>"

        "<h3>Offensive Typing</h3>"

        "<p>To Do</p>"

     )

    return HTML(raw_html)



displayPokemon()
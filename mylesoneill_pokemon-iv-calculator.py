# ======================================== #

# VARIABLES

# Edit these to change what pokemon you are dealing with



# --- Your Pokemon --- #



# Set the ID to be the pokemon you want to calculate

pokeID = 723



# Set the level of the pokemon (1 to 100)

level = 23



# EVs are learned skills, you can have a max of 255 for each stat but no more than 510 in total

EV = [24, 50, 30, 24, 24, 50]



# Current Stats

stats = [76, 52, 44, 38, 41, 39]



# Pokemon Natures can increase or decrease certain stats

nature = "Adamant"
# Lets Import Libraries and Read in the Data



import numpy as np

import pandas as pd

import math as math

from io import StringIO

from IPython.core.display import HTML



pokemon = pd.read_csv('../input/pokemon.csv')

natures = pd.read_csv('../input/natures.csv')
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



ourType1 = str(pokemon[pokemon['id'] == pokeID]['type1'].iloc[0]).lower()

ourType2 = str(pokemon[pokemon['id'] == pokeID]['type2'].iloc[0]).lower()
# Ok, lets calculate IVs



IV = [

    ((stats[0] - 10) * 100) / level - 2 * base[0] - math.floor(EV[0] / 4) - 100,

    ((stats[1]/nMod[1] - 5) * 100) / level - (2 * base[1]) - math.floor(EV[1] / 4),

    ((stats[2]/nMod[2] - 5) * 100) / level - (2 * base[2]) - math.floor(EV[2] / 4),

    ((stats[3]/nMod[3] - 5) * 100) / level - (2 * base[3]) - math.floor(EV[3] / 4),

    ((stats[4]/nMod[4] - 5) * 100) / level - (2 * base[4]) - math.floor(EV[4] / 4),

    ((stats[5]/nMod[5] - 5) * 100) / level - (2 * base[5]) - math.floor(EV[5] / 4)

]
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

                "<td>" + str(int(IV[0])) + "</td>"

                "<td>" + str(int(IV[1])) + "</td>"

                "<td>" + str(int(IV[2])) + "</td>"

                "<td>" + str(int(IV[3])) + "</td>"

                "<td>" + str(int(IV[4])) + "</td>"

                "<td>" + str(int(IV[5])) + "</td>"

            "</tr>"

        "</table>"

    )

    return HTML(raw_html)



displayPokemon()
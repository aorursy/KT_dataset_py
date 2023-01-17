zabt = "Papers on Medical Care"

znam = "Papers_on_Medical_Care"
zwds = "ability actions acute adapt adjunctive adjusted age allocation alternative application approaches ards arrest barriers boundaries capacity cardiac cardiomyopathy chain challenges characterization clia clinical communities control core course covid-19 crisis critical delivery determine disease distress ecmo efficiency elastomeric enhance equipment etiologies eua extracorporeal extrapulmonary faciitators facilitating frequency guidance hospital infected infection innovative interventions level manifestations manually masks maximize mechanical medical medications membrane methods mobilization mortality n95 natural nursing oral organ organization outcome outcomes overwhelmed oxygen oxygenation particularly patients payment people potentially practices prevention processes production protection protective regulatory resources respirators respiratory risk scarce shortages sick standards steroids supply support supportive surge syndrome technologies telemedicine thousands transmission trials usability ventilation viral virus workforce"
import os

import pandas as pd

import json

from IPython.core.display import display, HTML

# !pip uninstall spacy # Uncomment this if installed version of spacy fails.

# !pip install spacy # Uncomment this if spacy package is not installed.

import spacy

# !python -m spacy download en # Uncomment this if en language is not loaded in spacy package. 

nlp = spacy.load('en')
zchk = nlp(zwds)
ztop = '/kaggle/input/CORD-19-research-challenge'
zdf0 = pd.DataFrame(columns=['Folder', 'File', 'Match'])
%%capture



for zsub, zdir, zfis in os.walk(ztop):



    for zfil in zfis:

        if zfil.endswith(".json"):

            

            with open(zsub + os.sep + zfil) as f:

                zout = json.load(f)

            f.close()

            

            zout = " ".join([part['text'] for part in zout['abstract']])

            zout = zchk.similarity(nlp(zout))

            

            zdf0 = zdf0.append({'Folder': zsub.replace(ztop, ""), 'File': zfil, 'Match': zout}, ignore_index=True)

            

print(zdf0.head(4))
zdf0.to_csv(znam + '_Check.csv', index = False)
zdf6 = zdf0[zdf0.Match > 0.6].sort_values(by=['Match'], axis=0, ascending=False, inplace=False)

print(zdf6.head(4))
zdf6.to_csv(znam + '_Relevant.csv', index = False)
%%capture



zht0 = "<html>\n<head>\n"

zht0 = zht0 + "<title>Relevant Papers for Vaccines and Therapeutics</title>\n"

zht0 = zht0 + "<script>\nfunction openPop(x) {\nei = document.getElementById('pop_' + x);\n"

zht0 = zht0 + "ei.style.display='block';\nec = document.getElementsByClassName('pip');\nvar i;\n"

zht0 = zht0 + "for (i = 0; i < ec.length; i++) {\nif ( ec[i] != ei) { ec[i].style.display='none'; }; }; }\n"

zht0 = zht0 + "function shutPop(x) { document.getElementById('pop_' + x).style.display='none'; }\n</script>\n"

zht0 = zht0 + "<style>table, th, td { border: 1px solid black; }</style>\n"

zht0 = zht0 + "</head>\n<body>\n"

zht0 = zht0 + "<h1>" + zabt + "</h1>\n"

zht0 = zht0 + "<p>The following is a list of relevant papers.</p><br />\n"

zht0 = zht0 + "<p>Click on a Title to pop up its Abstract.</p><br />\n"

zht0 = zht0 + "<table>\n<tbody>\n<tr><th>Title</th>\n<th>Abstract</th></tr>\n"
zht6 = zht0 # zht6 is to be saved later as a file.

zhtd = zht0 # zhtd is a smaller version of zht6, for displaying in this notebook.



for indx, cont in zdf6.iterrows():

    

    with open(ztop + os.sep + cont['Folder'] + os.sep + cont['File']) as f:

        ztxt = json.load(f)

        f.close()

        

    ztxt = " ".join([part['text'] for part in ztxt['abstract']])

    

    zhta = "<tr><td><div onClick=openPop(" + str(indx) + ")>" + str(cont['File']) + "</div></td>\n"    

    zhta = zhta + "<td><div onClick=shutPop(" + str(indx) + ") class='pip' id='pop_" + str(indx) + "' style='display:none;'>" + ztxt + "</div></td></tr>\n"

    

    zht6 = zht6 + zhta

    if indx < 10:

        zhtd = zhtd + zhta



zht6 = zht6 + "</body>\n</html>"

zhtd = zhtd + "</body>\n</html>"
%%capture



zout = open(znam + "_Relevant_10.html","a")

zout.write(zht6)

zout.close()
display(HTML(zhtd))
zabt = "Papers on Diagnostics and Surveillance"

znam = "Papers_on_Diagnostics_and_Surveillance"
zwds = "able academic accelerator accessibility accuracy advanced aid analytics antibodies approach approaches area assay associated asymptomatic barriers bats bed-side bioinformatics biological capabilities capacity clinical coalition collecting communications companion convalescent coordination coupling covid-19 crispr cytokines demographic demographics denominators detection development devices diagnostic diagnostics disease distinguishing domestic drift efficacy efforts elisas entity environment environmental epidemic ethical evolution evolutionary execution experiments expertise exposure factors farmed food forces funding genetic genome genomics guidelines health hoc holistic host humans immediate impact important improve inclusive influenza information instruments intentional interventions issues laboratories latency legal leverage local locking longitudinal markers market mass measures mechanism migrate mitigate mitigation model mutations national naturally-occurring neutralizing non-profit occupational officials ongoing operational opportunities organism outcomes particular pathogen pathogens pcr people perspective platforms point-of-care policy potential practice predict preparedness private progression protocols public published purposes reagents recognizing recommendations recorded recruitment regions regulatory response risk roadmap samples sampling scale scaling schemes screening sector separation sequencing serosurveys sources species specific specificity spillover states streamlined sufficient supplies support surveillance swabs systematic tap target technology test testers testing therapeutic times track tradeoffs trafficked transmission understanding universities unknown variant viral virus widespread wildlife"
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
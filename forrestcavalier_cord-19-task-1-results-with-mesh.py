# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.core.display import display, HTML, Javascript

from string import Template

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import json, random

import IPython.display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


htmlTaskDetails="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>



<h2>CORD-19 Task Details</h2>

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=568">What is known about transmission, incubation, and environmental stability?</A>



<p><strong>What is known about transmission, incubation, and environmental stability?  What do we know about natural history, transmission, and diagnostics for the virus?  What have we learned about infection prevention and control?</strong></p>

<p>Specifically, we want to know what the literature reports about:</p>





<table class=l border=1>

<tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.<td>incubation (period OR periods OR 	times OR time)<td><A href="#task1a">Task1a Results below</a>

<tr><td>Prevalence of asymptomatic shedding and transmission (e.g., particularly children).<td>asymptomatic transmission<td><A href="#task1b">Task1b Results below</a>

<tr><td>Seasonality of transmission.<td>weather OR season OR seasonal<td><A href="#task1c">Task1c Results below</a>

<tr><td>Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).<td> inanimate OR surfaces OR disinfectants OR disinfection<td><A href="#task1d">Task1d Results below</a>

<tr><td>Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).<td>persistence OR stability<td><A href="#task1e">Task1e Results below</a>

<tr><td>Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).<td>materials OR metal OR plastic BEFORE 2020<td><A href="#task1f">Task1f Results below</a>

<tr><td>Natural history of the virus and shedding of it from an infected person<td>natural history (but the phrase natural history does not seem to be used the context of shedding.)  Also evolutionary<td><A href="#task1g">Task1g Results below</a>

<tr><td>Implementation of diagnostics and products to improve clinical processes<td>diagnostics<td><A href="#task1h">Task1h Results below</a>

<tr><td>Disease models, including animal models for infection, disease and transmission<td>model OR models<td><A href="#task1i">Task1i Results below</a>

<tr><td>Tools and studies to monitor phenotypic change and potential adaptation of the virus<td>variant OR adaptation OR phenotype OR genotype OR genetic OR genome OR strain<td><A href="#task1j">Task1j Results below</a>

<tr><td>Immune response and immunity<td>immune OR immunity OR immunoglobulin<td><A href="#task1k">Task1k Results below</a>

<tr><td>Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings<td>staff OR cross OR safety<td><A href="#task1l">Task1l Results below</a>

<tr><td>Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings<td>PPE or protective<td><A href="#task1m">Task1m Results below</a>

<tr><td>Role of the environment in transmission<td>tried: humidity transmission, temperature transmission, environment transmission.<td><A href="#task1n">Task1n Results below</a>

</table>

"""



h = display(HTML(htmlTaskDetails))
htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<hr><a name="task1a"><b>Task1a Kaggle Prompt:</b> Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.</a><p><b>Results:</b><p>

Searching for (incubation period OR periods OR times OR time) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...The median <b>incubation</b> <b>period</b> was 6.7 days, the interval time from between illness onset and seeing a doctor was 4.5 days.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31995857">

Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia.

</a>

<small>(PMID31995857</small>)

<br>...The mean <b>incubation</b> <b>period</b> was 5.2 days (95% confidence interval [CI], 4.1 to 7.0), with the 95th percentile of the distribution at 12.5 days.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>...Human-to-human transmissions have been described with <b>incubation</b> <b>times</b> between 2-10 days, facilitating its spread via droplets, contaminated hands or surfaces.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046816">

Effectiveness of airport screening at detecting travellers infected with novel coronavirus (2019-nCoV).

</a>

<small>(PMID32046816</small>)

<br>...In our baseline scenario, we estimated that 46% (95% confidence interval: 36 to 58) of infected travellers would not be detected, depending on <b>incubation</b> <b>period</b>, sensitivity of exit and entry screening, and proportion of asymptomatic cases.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046819">

Incubation period of 2019 novel coronavirus (2019-nCoV) infections among travellers from Wuhan, China, 20-28 January 2020.

</a>

<small>(PMID32046819</small>)

<br>...Using the travel history and symptom onset of 88 confirmed cases that were detected outside Wuhan in the early outbreak phase, we  estimate the mean <b>incubation</b> <b>period</b> to be 6.4 days (95% credible interval: 5.6-7.7), ranging from 2.1 to 11.1 days (2.5th to 97.5th percentile).

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period+OR+periods+OR+times+OR+time&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...The disease is transmitted by inhalation or contact with infected droplets and the <b>incubation</b> <b>period</b> ranges from 2 to 14 d.

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

</table>

<p>There are also 199 matches before 2019/12

<hr><a name="task1b"><b>Task1b Kaggle Prompt:</b> Prevalence of asymptomatic shedding and transmission (e.g., particularly children).</a><p><b>Results:</b><p>

Searching for (asymptomatic transmission) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=asymptomatic+transmission&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=asymptomatic+transmission&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240128">

Asymptomatic and Presymptomatic SARS-CoV-2 Infections in Residents of a Long-Term Care Skilled Nursing Facility - King County, Washington, March 2020.

</a>

<small>(PMID32240128</small>)

<br>...The reverse transcription-polymerase chain reaction (RT-PCR) testing cycle threshold (Ct) values indicated large quantities  of viral RNA in <b>asymptomatic</b>, presymptomatic, and symptomatic residents, suggesting the potential for <b>transmission</b> regardless of symptoms.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32003551">

Transmission of 2019-nCoV Infection from an Asymptomatic Contact in Germany.

</a>

<small>(PMID32003551</small>)

<br>....

<td>Case Reports; Letter</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 103 matches before 2019/12

<hr><a name="task1c"><b>Task1c Kaggle Prompt:</b> Seasonality of transmission.</a><p><b>Results:</b><p>

Searching for (weather OR season OR seasonal) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/C/CH/Chiroptera">Chiroptera</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710904">

The complex associations of climate variability with seasonal influenza A and B virus transmission in subtropical Shanghai, China.

</a>

<small>(PMID31710904</small>)

<br>...Most previous studies focused on the association between climate variables and <b>seasonal</b> influenza activity in tropical or temperate zones, little is known about the associations in different influenza types in subtropical China...We suggest the careful use of meteorological variables in influenza prediction in subtropical regions, considering such complex associations, which may facilitate  government and health authorities to better minimize the impacts of <b>seasonal</b> influenza..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31989546">

Epidemic Influenza Seasons from 2008 to 2018 in Poland: A Focused Review of Virological Characteristics.

</a>

<small>(PMID31989546</small>)

<br>...The peak incidence of influenza infections has regularly been in January-March each epidemic <b>season</b>...The number of tested specimens ranged from 2066 to 8367 per <b>season</b> from 2008/2009 to 2017/2018...Type A virus predominated in nine out of the ten seasons and type B virus of the Yamagata lineage in the 2017/2018 <b>season</b>...There was a sharp increase in the proportion of laboratory confirmations of influenza infection from <b>season</b> to season in relation to the number of specimens examined,  from 3.2% to 42.4% over the decade...The number of confirmations, enabling a prompt commencement of antiviral treatment, related to the number of specimens collected from patients and on the virological situation in a given <b>season</b>.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832902">

Virological and Epidemiological Situation in the Influenza Epidemic Seasons 2016/2017 and 2017/2018 in Poland.

</a>

<small>(PMID31832902</small>)

<br>...In the 2016/2017 epidemic <b>season</b> in Poland, the incidence of influenza was 1,692 per 100,000 population...The influenza A virus, subtype A/H3N2/, was the predominant one in that <b>season</b>...However, in the most recent 2017/2018 epidemic <b>season</b>, the incidence exceeded 1,782 per 100,000 already by August of 2018...In this <b>season</b>, influenza B virus predominated, while the A/H1N1/pdm09 strain was most frequent among the influenza A subtypes...As of the 2017/2018 <b>season</b>, a quadrivalent vaccine, consisting of two antigens of influenza A subtypes and another two of influenza B virus, was available in Poland.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31823763">

Evaluation of the influenza sentinel surveillance system in the Democratic Republic of Congo, 2012-2015.

</a>

<small>(PMID31823763</small>)

<br>...However, due to limited resources no actions were undertaken to mitigate the impact of <b>seasonal</b> influenza epidemics.

<td>Evaluation Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31004768">

Respiratory virus infection among hospitalized adult patients with or without clinically apparent respiratory infection: a prospective cohort study.

</a>

<small>(PMID31004768</small>)

<br>...METHODS: This prospective cohort study was conducted during the 2018 winter influenza <b>season</b>.

<td>Comparative Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31489496">

Paramyxoviruses respiratory syncytial virus, parainfluenza virus, and human metapneumovirus infection in pediatric hospitalized patients and climate correlation in a subtropical region of southern China: a 7-year survey.

</a>

<small>(PMID31489496</small>)

<br>...RSV and HMPV had similar <b>seasonal</b> patterns, with two prevalence peaks every year.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32160889">

Geographical tracking and mapping of coronavirus disease COVID-19/severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) epidemic and associated events around the world: how 21st century GIS technologies are supporting the global fight against outbreaks and epidemics.

</a>

<small>(PMID32160889</small>)

<br>...As with the  original SARS-CoV epidemic of 2002/2003 and with <b>seasonal</b> influenza, geographic information systems and methods, including, among other application possibilities, online real-or near-real-time mapping of disease cases and of social media reactions to disease spread, predictive risk mapping using population travel data, and tracing and mapping super-spreader trajectories and contacts across space and time, are proving indispensable for timely and effective epidemic monitoring and response.

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513753">

A serological survey of canine respiratory coronavirus in New Zealand.

</a>

<small>(PMID31513753</small>)

<br>...Differences in CRCoV seroprevalence between regions and lack of <b>seasonal</b> pattern indicate that factors other than external temperatures may be important in the epidemiology of CRCoV in New Zealand.Clinical relevance: Our data suggest  that CRCoV should be included in investigations of cases of infectious canine tracheobronchitis, particularly if these occur among dogs vaccinated with current vaccines, which do not include CRCoV antigens..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=weather+OR+season+OR+seasonal&from=CORD19#/C/CH/Chiroptera">Chiroptera</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31910439">

Mating strategy is determinant of adenovirus prevalence in European bats.

</a>

<small>(PMID31910439</small>)

<br>...kuhlii and Nyctalus lasiopterus and we found that in the latter, males were more likely to be infected by adenoviruses than females, due to the immunosuppressing consequence of testosterone during the mating <b>season</b>.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

</table>

<p>There are also 955 matches before 2019/12

<hr><a name="task1d"><b>Task1d Kaggle Prompt:</b> Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).</a><p><b>Results:</b><p>

Searching for (inanimate OR surfaces OR disinfectants OR disinfection) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=inanimate+OR+surfaces+OR+disinfectants+OR+disinfection&from=CORD19#/D/DI/Disinfectants">Disinfectants</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=inanimate+OR+surfaces+OR+disinfectants+OR+disinfection&from=CORD19#/C/CO/Collectins">Collectins</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=inanimate+OR+surfaces+OR+disinfectants+OR+disinfection&from=CORD19#/D/DI/Disinfectants">Disinfectants</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>...Human-to-human transmissions have been described with incubation times between 2-10 days, facilitating its spread via droplets, contaminated hands or <b>surfaces</b>... We therefore reviewed the literature on all available information about the persistence of human and veterinary coronaviruses on <b>inanimate</b> surfaces as well as inactivation strategies with biocidal agents used for chemical disinfection, e.g...The analysis of 22 studies reveals that human coronaviruses such as Severe Acute Respiratory Syndrome (SARS) coronavirus, Middle East Respiratory Syndrome (MERS) coronavirus or endemic human coronaviruses (HCoV) can persist on <b>inanimate</b> surfaces like metal, glass or plastic for up to 9 days, but can be efficiently inactivated by surface disinfection procedures with 62-71% ethanol, 0.5% hydrogen peroxide or 0.1% sodium hypochlorite within 1 minute.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=inanimate+OR+surfaces+OR+disinfectants+OR+disinfection&from=CORD19#/C/CO/Collectins">Collectins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152944">

Collectins: Innate Immune Pattern Recognition Molecules.

</a>

<small>(PMID32152944</small>)

<br>...Collectins can  be found in serum as well as in a range of tissues at the mucosal <b>surfaces</b>.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

</table>

<p>There are also 415 matches before 2019/12

<hr><a name="task1e"><b>Task1e Kaggle Prompt:</b> Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).</a><p><b>Results:</b><p>

Searching for (persistence OR stability) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/P/PO/Polymerase Chain Reaction">Polymerase Chain Reaction</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>... We therefore reviewed the literature on all available information about the <b>persistence</b> of human and veterinary coronaviruses on inanimate surfaces as well as inactivation strategies with biocidal agents used for chemical disinfection, e.g.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31931793">

An evaluation of the Zambia influenza sentinel surveillance system, 2011-2017.

</a>

<small>(PMID31931793</small>)

<br>...METHODS: We used the Centers for Disease Control and Prevention guidelines to evaluate the performance of the influenza surveillance system (ISS) in Zambia during 2011-2017 using 9 attributes: (i) data quality and completeness, (ii) timeliness, (iii) representativeness, (iv) flexibility, (v) simplicity, (vi) acceptability, (vii) <b>stability</b>, (viii) utility, and (ix) sustainability...Key strengths of the system were the quality of data generated (score: 2.9), its flexibility (score: 3.0) especially to monitor viral pathogens other than influenza viruses, its simplicity (score: 2.8), acceptability (score: 3.0) and <b>stability</b> (score: 2.6) over the review period and  its relatively low cost ($310,000 per annum).

<td>Evaluation Study; Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31823763">

Evaluation of the influenza sentinel surveillance system in the Democratic Republic of Congo, 2012-2015.

</a>

<small>(PMID31823763</small>)

<br>...The performance of the system was evaluated using eight surveillance attributes: (i)  data quality and completeness for key variables, (ii) timeliness, (iii) representativeness, (iv) flexibility, (v) simplicity, (vi) acceptability, (vii) <b>stability</b> and (viii) utility...Other strengths of the system were timeliness, simplicity,  <b>stability</b> and utility that scored > 70% each...The simplicity of the system contributed to its <b>stability</b>.

<td>Evaluation Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...The investigation of virus detection and <b>persistence</b> in semen across a range of emerging viruses is useful for clinical and public health reasons, in particular for viruses that lead to high mortality or morbidity rates or to epidemics.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/P/PO/Polymerase Chain Reaction">Polymerase Chain Reaction</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31629228">

Thermally stable and uniform DNA amplification with picosecond laser ablated graphene rapid thermal cycling device.

</a>

<small>(PMID31629228</small>)

<br>...A thin-film electrode with the aforementioned MLG as the heater was demonstrated to significantly enhance temperature <b>stability</b> for each stage of the thermal cycle.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=persistence+OR+stability&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...In vitro biochemical data further suggest that mutations at these two sites affect RNA binding, EC <b>stability</b>, but not polymerase catalytic rate (kcat)  and apparent NTP affinity (KM,NTP).

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 1061 matches before 2019/12

<hr><a name="task1f"><b>Task1f Kaggle Prompt:</b> Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).</a><p><b>Results:</b><p>

Searching for (materials OR metal OR plastic) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/P/PU/Public Health">Public Health</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/D/DE/Dengue">Dengue</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/A/AU/Autophagy">Autophagy</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119961">

Anti-HCV, nucleotide inhibitors, repurposing against COVID-19.

</a>

<small>(PMID32119961</small>)

<br>...<b>MATERIALS</b> AND METHODS: In this study, sequence analysis, modeling, and docking are used to build a model for Wuhan COVID-19 RdRp.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216961">

Imaging manifestations and diagnostic value of chest CT of coronavirus disease 2019 (COVID-19) in the Xiaogan area.

</a>

<small>(PMID32216961</small>)

<br>...<b>MATERIALS</b> AND METHODS: The complete clinical and imaging data of 114 confirmed COVID-19 patients treated in  Xiaogan Hospital were analysed retrospectively.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32100485">

Chest Radiographic and CT Findings of the 2019 Novel Coronavirus Disease (COVID-19): Analysis of Nine Patients Treated in Korea.

</a>

<small>(PMID32100485</small>)

<br>...<b>MATERIALS</b> AND METHODS: As part of a multi-institutional collaboration coordinated by the Korean Society of Thoracic Radiology, we collected nine patients with COVID-19 infections who had undergone  chest radiography and CT scans.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32109443">

Clinical and computed tomographic imaging features of novel coronavirus pneumonia caused by SARS-CoV-2.

</a>

<small>(PMID32109443</small>)

<br>...<b>MATERIALS</b> AND METHODS: A retrospective analysis was performed on the imaging findings of patients confirmed with COVID-19 pneumonia who had chest CT scanning  and treatment after disease onset.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>...The analysis of 22 studies reveals that human coronaviruses such as Severe Acute Respiratory Syndrome (SARS) coronavirus, Middle East Respiratory Syndrome (MERS) coronavirus or endemic human coronaviruses (HCoV) can persist on inanimate surfaces like <b>metal</b>, glass or plastic for up to 9 days, but can be efficiently inactivated by surface disinfection procedures with 62-71% ethanol, 0.5% hydrogen peroxide or 0.1% sodium hypochlorite within 1 minute.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/P/PU/Public Health">Public Health</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...GHD/EMPHNET has the scientific expertise to contribute to elevating the level of country alert and preparedness in the EMR and to provide technical support through health promotion, training and training <b>materials</b>, guidelines, coordination, and communication...The FETPs are currently actively participating in surveillance and screening at the ports of entry, development of communication <b>materials</b> and guidelines, and sharing information to health professionals and the public.

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/D/DE/Dengue">Dengue</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951602">

Validation of the easyscreen flavivirus dengue alphavirus detection kit based on  3base amplification technology and its application to the 2016/17 Vanuatu dengue  outbreak.

</a>

<small>(PMID31951602</small>)

<br>...METHODOLOGY/PRINCIPAL FINDING: Synthetic constructs, viral nucleic acids, intact viral particles and characterised reference <b>materials</b> were  used to determine the specificity and sensitivity of the assays.

<td>Journal Article; Validation Study</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32083983">

Molecular characterization of Brazilian wild-type strains of bovine respiratory syncytial virus reveals genetic diversity and a putative new subgroup of the virus.

</a>

<small>(PMID32083983</small>)

<br>...Background: Bovine orthopneumovirus, formerly known as bovine respiratory syncytial virus (BRSV), is frequently associated with bovine respiratory disease  (BRD).Aim: To perform the molecular characterization of the G and F proteins of Brazilian wild-type BRSV strains derived from bovine respiratory infections in both beef and dairy cattle.<b>Materials</b> and Methods: Ten BRSV strains derived from a dairy heifer rearing unit (n = 3) in 2011 and steers of three other feedlots (n = 7) in 2014 and 2015 were analyzed.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=materials+OR+metal+OR+plastic&from=CORD19#/A/AU/Autophagy">Autophagy</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31433241">

Long non-coding RNA MEG3 attends to morphine-mediated autophagy of HT22 cells through modulating ERK pathway.

</a>

<small>(PMID31433241</small>)

<br>...<b>Materials</b> and methods: HT22 cells were subjected to 10 microM morphine for 24 h.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 491 matches before 2019/12

<hr><a name="task1g"><b>Task1g Kaggle Prompt:</b> Natural history of the virus and shedding of it from an infected person</a><p><b>Results:</b><p>

Searching for (natural history OR evolutionary) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/A/AT/Atherosclerosis">Atherosclerosis</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/C/CR/Croup">Croup</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31536759">

Comprehensive codon usage analysis of porcine deltacoronavirus.

</a>

<small>(PMID31536759</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/A/AT/Atherosclerosis">Atherosclerosis</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31925708">

From Endothelium to Lipids, Through microRNAs and PCSK9: A Fascinating Travel Across Atherosclerosis.

</a>

<small>(PMID31925708</small>)

<br>...Recently, proprotein convertase subtilisin/kexin type 9 (PCSK9) has  been recognized as a fundamental regulator of LDL-C and anti-PCSK9 monoclonal antibodies have been approved for therapeutic use in hypercholesterolemia, with the promise to subvert the <b>natural</b> <b>history</b> of the disease.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=natural+history+OR+evolutionary&from=CORD19#/C/CR/Croup">Croup</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542653">

Defining atypical croup: A case report and review of the literature.

</a>

<small>(PMID31542653</small>)

<br>...CONCLUSIONS: Atypical croup is a poorly defined clinical entity that is used to describe recurrent, refractory, or croup-like illness that follows an uncharacteristic <b>natural</b> <b>history</b>.

<td>Case Reports; Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 210 matches before 2019/12

<hr><a name="task1h"><b>Task1h Kaggle Prompt:</b> Implementation of diagnostics and products to improve clinical processes</a><p><b>Results:</b><p>

Searching for (diagnostics) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=diagnostics&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=diagnostics&from=CORD19#/R/RE/Real-Time Polymerase Chain Reaction">Real-Time Polymerase Chain Reaction</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=diagnostics&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32036774">

Emerging novel coronavirus (2019-nCoV)-current scenario, evolutionary perspective based on genome analysis and recent developments.

</a>

<small>(PMID32036774</small>)

<br>...The successful virus isolation attempts have made doors open for developing better <b>diagnostics</b> and effective vaccines helping in combating the spread of the virus to newer areas..

<td>Journal Article; Review</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32007627">

Novel coronavirus: From discovery to clinical diagnostics.

</a>

<small>(PMID32007627</small>)

<br>...We have described the discovery, emergence, genomic characteristics, and clinical <b>diagnostics</b> of 2019-nCoV..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32027631">

Initial Public Health Response and Interim Clinical Guidance for the 2019 Novel Coronavirus Outbreak - United States, December 31, 2019-February 4, 2020.

</a>

<small>(PMID32027631</small>)

<br>...Although these measures might not prevent the eventual establishment of ongoing, widespread transmission of the virus in the United States, they are being implemented to 1) slow the spread of illness; 2) provide time to better prepare health care systems and the general public to be ready if widespread transmission with substantial associated illness occurs; and 3) better characterize 2019-nCoV infection to guide public health recommendations and the development of medical countermeasures including <b>diagnostics</b>, therapeutics, and vaccines.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=diagnostics&from=CORD19#/R/RE/Real-Time Polymerase Chain Reaction">Real-Time Polymerase Chain Reaction</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32156330">

Rapid establishment of laboratory diagnostics for the novel coronavirus SARS-CoV-2 in Bavaria, Germany, February 2020.

</a>

<small>(PMID32156330</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 481 matches before 2019/12

<hr><a name="task1i"><b>Task1i Kaggle Prompt:</b> Disease models, including animal models for infection, disease and transmission</a><p><b>Results:</b><p>

Searching for (model OR models) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/P/PE/Peptidyl-Dipeptidase A">Peptidyl-Dipeptidase A</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/M/MA/Macrophages">Macrophages</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/S/SI/Signal Transduction">Signal Transduction</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/A/AD/Adjuvants, Immunologic">Adjuvants, Immunologic</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/E/EP/Epidemics">Epidemics</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=14><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119961">

Anti-HCV, nucleotide inhibitors, repurposing against COVID-19.

</a>

<small>(PMID32119961</small>)

<br>...MATERIALS AND METHODS: In this study, sequence analysis, modeling, and docking are used to build a <b>model</b> for Wuhan COVID-19 RdRp...Additionally, the newly emerged Wuhan HCoV RdRp <b>model</b> is targeted by anti-polymerase drugs, including the approved drugs Sofosbuvir and Ribavirin...SIGNIFICANCE: The present study presents a perfect <b>model</b> for COVID-19 RdRp enabling its testing in silico against anti-polymerase drugs.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119825">

Feasibility of controlling COVID-19 outbreaks by isolation of cases and contacts.

</a>

<small>(PMID32119825</small>)

<br>...Here we use a mathematical <b>model</b> to assess if isolation and contact tracing are able to control onwards transmission from imported cases of COVID-19...METHODS: We developed a stochastic transmission <b>model</b>, parameterised to the COVID-19 outbreak...We used the <b>model</b> to quantify the  potential effectiveness of contact tracing and isolation of cases at controlling  a severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)-like pathogen...We assumed isolation prevented all further transmission in the <b>model</b>...This <b>model</b> can be modified to reflect updated transmission characteristics and more specific definitions of outbreak control to assess the potential success of local response efforts.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238182">

The fiscal value of human lives lost from coronavirus disease (COVID-19) in China.

</a>

<small>(PMID32238182</small>)

<br>...Re-estimation of the economic <b>model</b> alternately with 5% and 10 discount rates led to a reduction in the expected total fiscal value by 21.3% and 50.4%, respectively...Furthermore, the re-estimation of the economic <b>model</b> using the world's highest average life expectancy of 87.1 years (which is that of Japanese  females), instead of the national life expectancy of 76.4 years, increased the total fiscal value by Int$ 229,456,430 (24.8%)..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32167747">

Evidence of the COVID-19 Virus Targeting the CNS: Tissue Distribution, Host-Virus Interaction, and Proposed Neurotropic Mechanisms.

</a>

<small>(PMID32167747</small>)

<br>...Also, we debate the need for a <b>model</b> for staging COVID-19 based on neurological tissue involvement..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...Remdesivir (GS-5734) effectively inhibited MERS coronavirus (MERS-CoV) replication in vitro, and showed efficacy against Severe Acute Respiratory Syndrome (SARS)-CoV in a mouse <b>model</b>...Here, we tested the efficacy of prophylactic and therapeutic remdesivir treatment in a nonhuman primate <b>model</b> of  MERS-CoV infection, the rhesus macaque.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32182811">

Reverse Logistics Network Design for Effective Management of Medical Waste in Epidemic Outbreaks: Insights from the Coronavirus Disease 2019 (COVID-19) Outbreak in Wuhan (China).

</a>

<small>(PMID32182811</small>)

<br>...The application of the <b>model</b> is illustrated with a case study based on the outbreak of the coronavirus disease 2019 (COVID-19) in Wuhan, China.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32189081">

AI-Driven Tools for Coronavirus Outbreak: Need of Active Learning and Cross-Population Train/Test Models on Multitudinal/Multimodal Data.

</a>

<small>(PMID32189081</small>)

<br>...However, unlike other healthcare issues, for COVID-19, to detect COVID-19, AI-driven tools are expected to have active learning-based cross-population train/test <b>models</b> that employs multitudinal and multimodal data, which is the primary purpose of the paper..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32092911">

Rigidity of the Outer Shell Predicted by a Protein Intrinsic Disorder Model Sheds Light on the COVID-19 (Wuhan-2019-nCoV) Infectivity.

</a>

<small>(PMID32092911</small>)

<br>....

<td>Editorial</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32111262">

A mathematical model for simulating the phase-based transmissibility of a novel coronavirus.

</a>

<small>(PMID32111262</small>)

<br>...This study aimed to develop a mathematical <b>model</b> for calculating the transmissibility of the virus...METHODS: In this study, we developed a Bats-Hosts-Reservoir-People transmission network <b>model</b> for simulating the potential transmission from the infection source (probably be  bats) to the human infection...Since the Bats-Hosts-Reservoir network was hard to  explore clearly and public concerns were focusing on the transmission from Huanan Seafood Wholesale Market (reservoir) to people, we simplified the <b>model</b> as Reservoir-People (RP) transmission network model...The next generation matrix approach was adopted to calculate the basic reproduction number (R0) from the RP  <b>model</b> to assess the transmissibility of the SARS-CoV-2...CONCLUSIONS: Our <b>model</b> showed that the transmissibility of SARS-CoV-2 was higher than the Middle East respiratory syndrome in the Middle East countries, similar to severe acute respiratory syndrome, but lower than MERS in the Republic of Korea..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32091395">

Estimated effectiveness of symptom and risk screening to prevent the spread of COVID-19.

</a>

<small>(PMID32091395</small>)

<br>...Previously, we developed a mathematical <b>model</b> to understand factors governing the effectiveness of traveller screening to prevent spread of emerging pathogens (Gostic et al., 2015).

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32007143">

Epidemiological and clinical characteristics of 99 cases of 2019 novel coronavirus pneumonia in Wuhan, China: a descriptive study.

</a>

<small>(PMID32007143</small>)

<br>...In general, characteristics of patients who died were in line with the MuLBSTA score, an early warning <b>model</b> for predicting mortality in viral pneumonia.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513753">

A serological survey of canine respiratory coronavirus in New Zealand.

</a>

<small>(PMID31513753</small>)

<br>...Age of dog, sampling month, region, and presence of abnormal respiratory signs were included in the initial logistic regression <b>model</b>.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231374">

Data-based analysis, modelling and forecasting of the COVID-19 outbreak.

</a>

<small>(PMID32231374</small>)

<br>...On the basis of a Susceptible-Infectious-Recovered-Dead (SIDR) <b>model</b>, we provide estimations of the basic reproduction number (R0), and the per day infection mortality and recovery  rates...By calibrating the parameters of the SIRD <b>model</b> to the reported data, we also attempt to forecast the evolution of the outbreak at the epicenter three weeks ahead, i.e.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710904">

The complex associations of climate variability with seasonal influenza A and B virus transmission in subtropical Shanghai, China.

</a>

<small>(PMID31710904</small>)

<br>...Generalized linear <b>models</b> (GLMs), distributed lag non-linear models (DLNMs) and regression tree models were developed to assess such associations.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31505263">

Influenza A virus infection induces liver injury in mice.

</a>

<small>(PMID31505263</small>)

<br>...This study aimed to investigate the occurrence of hepatitis by establishing a <b>model</b> for infected mice for three different subtypes of respiratory IAVs (H1N1, H5N1, and H7N2)...All these data showed that the mouse <b>model</b> suitably contributed valuable information about the mechanism underlying the occurrence of hepatitis induced by respiratory influenza virus..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31489496">

Paramyxoviruses respiratory syncytial virus, parainfluenza virus, and human metapneumovirus infection in pediatric hospitalized patients and climate correlation in a subtropical region of southern China: a 7-year survey.

</a>

<small>(PMID31489496</small>)

<br>...Multiple linear regression <b>models</b> were established for RSV, PIV, and HMPV prevalence and meteorological factors (p < 0.05).

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/P/PE/Peptidyl-Dipeptidase A">Peptidyl-Dipeptidase A</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209118">

Using the spike protein feature to predict infection risk and monitor the evolutionary dynamic of coronavirus.

</a>

<small>(PMID32209118</small>)

<br>...In this study, a prediction <b>model</b> is proposed to evaluate the infection risk of non-human-origin coronavirus for early warning...To capture the key information of the spike protein, three feature encoding algorithms (amino acid composition, AAC; parallel correlation-based pseudo-amino-acid composition, PC-PseAAC and G-gap dipeptide composition, GGAP) were used to train 41 random forest <b>models</b>...The predictive <b>model</b> achieved the maximum ACC of 98.18% coupled with the Matthews correlation coefficient (MCC) of 0.9638.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/M/MA/Macrophages">Macrophages</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31825972">

IL-4/IL-13 polarization of macrophages enhances Ebola virus glycoprotein-dependent infection.

</a>

<small>(PMID31825972</small>)

<br>...METHODS/MAIN FINDINGS: We utilized a BSL2 EBOV <b>model</b> virus, infectious, recombinant vesicular stomatitis virus encoding EBOV glycoprotein (GP) (rVSV/EBOV GP) in place of its native glycoprotein...In vivo IL-4/IL-13 administration significantly increased virus-mediated mortality in a mouse <b>model</b> and transfer of ex vivo IL-4/IL-13-treated murine peritoneal macrophages into the peritoneal cavity of mice enhanced pathogenesis.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/S/SI/Signal Transduction">Signal Transduction</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31758397">

Methyltransferase of a cell culture-adapted hepatitis E inhibits the MDA5 receptor signaling pathway.

</a>

<small>(PMID31758397</small>)

<br>...As a mouse <b>model</b> is not available, a recent development of a cell culture-adapted HEV strain (47832c) is  considered as a very important tools for molecular analysis of HEV pathogenesis in cells.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/A/AD/Adjuvants, Immunologic">Adjuvants, Immunologic</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31856726">

Prediction of novel mouse TLR9 agonists using a random forest approach.

</a>

<small>(PMID31856726</small>)

<br>...Therefore, we developed a cross-validated ensemble classifier of 20 random forest <b>models</b>...Predictions on 6000 randomly generated ODNs were ranked and the top 100 ODNs were synthesized and experimentally tested for activity in a mTLR9 reporter cell assay, with 91 of the 100 selected ODNs showing high activity, confirming the accuracy of the <b>model</b> in  predicting mTLR9 activity.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31753544">

UV-C irradiation is able to inactivate pathogens found in commercially collected  porcine plasma as demonstrated by swine bioassay.

</a>

<small>(PMID31753544</small>)

<br>...However, the final validation of the UV-C light as safety feature should be conducted with commercial liquid plasma and using the pig bioassay <b>model</b>.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/E/EP/Epidemics">Epidemics</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31905206">

Risk perception and behavioral change during epidemics: Comparing models of individual and collective learning.

</a>

<small>(PMID31905206</small>)

<br>...Methodological approaches that range from purely physics-based diffusion <b>models</b> to data-driven environmental methods rely on agent-based modeling to accommodate context-dependent learning and social interactions in a diffusion process...The differences between collective learning and individual learning have not been sufficiently explored in diffusion modeling in general and in agent-based <b>models</b> of socio-environmental systems in particular...To address this research gap, we explored the implications of intelligent learning on the gradient from individual to collective learning, using an agent-based <b>model</b> enhanced by machine learning...The choice of how to represent social learning in an agent-based <b>model</b> could be driven by existing cultural and social norms prevalent in a modeled society..

<td>Comparative Study; Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152943">

CLEC5A: A Promiscuous Pattern Recognition Receptor to Microbes and Beyond.

</a>

<small>(PMID32152943</small>)

<br>...For example, in vivo studies in mouse <b>models</b> have demonstrated that CLEC5A is responsible for flaviviruses-induced hemorrhagic shock and neuroinflammation, and a CLEC5A polymorphism in humans is associated with disease severity following infection with dengue virus.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=model+OR+models&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31733447">

Deiminated proteins in extracellular vesicles and serum of llama (Lama glama)-Novel insights into camelid immunity.

</a>

<small>(PMID31733447</small>)

<br>...1758) as a <b>model</b> animal.

<td>Journal Article; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 4617 matches before 2019/12

<hr><a name="task1j"><b>Task1j Kaggle Prompt:</b> Tools and studies to monitor phenotypic change and potential adaptation of the virus</a><p><b>Results:</b><p>

Searching for (variant OR adaptation OR phenotype OR genotype OR genetic OR genome OR strain) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SA/SARS Virus">SARS Virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SW/Swine Diseases">Swine Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CH/Chiroptera">Chiroptera</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/D/DO/Dog Diseases">Dog Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/H/HI/HIV-1">HIV-1</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/D/DE/Dengue Virus">Dengue Virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RE/Respiratory Syncytial Virus Infections">Respiratory Syncytial Virus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/M/MA/Macrophages">Macrophages</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RE/Real-Time Polymerase Chain Reaction">Real-Time Polymerase Chain Reaction</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RN/RNA Replicase">RNA Replicase</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Camelus">Camelus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Cattle">Cattle</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/H/HE/Hepatitis E virus">Hepatitis E virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SP/Spike Glycoprotein, Coronavirus">Spike Glycoprotein, Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EN/Endothelial Cells">Endothelial Cells</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EX/Extracellular Vesicles">Extracellular Vesicles</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/O/OR/Oropharynx">Oropharynx</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CE/Cell Line">Cell Line</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=25><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32036774">

Emerging novel coronavirus (2019-nCoV)-current scenario, evolutionary perspective based on genome analysis and recent developments.

</a>

<small>(PMID32036774</small>)

<br>...The <b>genetic</b> analyses predict bats as the most probable source of 2019-nCoV though further investigations needed to confirm the origin of the novel virus.

<td>Journal Article; Review</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32081569">

Community pharmacist in public health emergencies: Quick to action against the coronavirus 2019-nCoV outbreak.

</a>

<small>(PMID32081569</small>)

<br>...The 2019-nCoV infection that is caused by a novel <b>strain</b> of coronavirus was first detected in China in the end of December 2019 and declared a public health emergency of international concern by the World Health Organization on January 30, 2020.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32031583">

Molecular Diagnosis of a Novel Coronavirus (2019-nCoV) Causing an Outbreak of Pneumonia.

</a>

<small>(PMID32031583</small>)

<br>...METHODS: We developed two 1-step quantitative real-time reverse-transcription PCR assays to detect two different regions (ORF1b and N) of the viral <b>genome</b>.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32174053">

False-Negative Results of Real-Time Reverse-Transcriptase Polymerase Chain Reaction for Severe Acute Respiratory Syndrome Coronavirus 2: Role of Deep-Learning-Based CT Diagnosis and Insights from Two Cases.

</a>

<small>(PMID32174053</small>)

<br>...The  nucleic acid test or <b>genetic</b> sequencing serves as the gold standard method for confirmation of infection, yet several recent studies have reported false-negative results of real-time reverse-transcriptase polymerase chain reaction (rRT-PCR).

<td>Case Reports</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209231">

Computers and viral diseases. Preliminary bioinformatics studies on the design of a synthetic vaccine and a preventative peptidomimetic antagonist against the SARS-CoV-2 (2019-nCoV, COVID-19) coronavirus.

</a>

<small>(PMID32209231</small>)

<br>...This paper concerns study of the <b>genome</b> of the Wuhan Seafood Market isolate believed to represent the causative agent of the disease COVID-19.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32200634">

Protein Structure and Sequence Reanalysis of 2019-nCoV Genome Refutes Snakes as Its Intermediate Host and the Unique Similarity between Its Spike Protein Insertions and HIV-1.

</a>

<small>(PMID32200634</small>)

<br>...Next, using metagenomic samples from Manis javanica, we assembled a draft <b>genome</b> of the 2019-nCoV-like coronavirus, which shows 73% coverage and 91% sequence identity to the 2019-nCoV genome.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31950289">

Recombinant adenovirus carrying a core neutralizing epitope of porcine epidemic diarrhea virus and heat-labile enterotoxin B of Escherichia coli as a mucosal vaccine.

</a>

<small>(PMID31950289</small>)

<br>...Moreover, a cell-mediated immune response was promoted  in immunized mice, and the neutralizing antibody inhibited both the vaccine <b>strain</b> and the emerging PEDV isolate.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32048560">

The global spread of 2019-nCoV: a molecular evolutionary analysis.

</a>

<small>(PMID32048560</small>)

<br>...A Maximum Clade Credibility tree has been built using the 29 available whole <b>genome</b> sequences of 2019-nCoV and two whole genome sequences that are highly similar sequences from Bat SARS-like Coronavirus available in GeneBank...Moreover, our study describes the same population <b>genetic</b> dynamic underlying the  SARS 2003 epidemic, and suggests the urgent need for the development of effective molecular surveillance strategies of Betacoronavirus among animals and Rhinolophus of the bat family..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32015507">

A pneumonia outbreak associated with a new coronavirus of probable bat origin.

</a>

<small>(PMID32015507</small>)

<br>...Full-length <b>genome</b> sequences were obtained from five patients at an early stage of the outbreak.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32015508">

A new coronavirus associated with human respiratory disease in China.

</a>

<small>(PMID32015508</small>)

<br>...Metagenomic RNA sequencing(4) of a sample of bronchoalveolar lavage fluid from the patient identified a new RNA virus <b>strain</b> from the family Coronaviridae, which is designated here 'WH-Human 1' coronavirus  (and has also been referred to as '2019-nCoV')...Phylogenetic analysis of the complete viral <b>genome</b> (29,903 nucleotides) revealed that the virus was most closely related (89.1% nucleotide similarity) to a group of SARS-like coronaviruses (genus Betacoronavirus, subgenus Sarbecovirus) that had previously  been found in bats in China(5).

<td>Case Reports; Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32094336">

High expression of ACE2 receptor of 2019-nCoV on the epithelial cells of oral mucosa.

</a>

<small>(PMID32094336</small>)

<br>...To investigate the potential route of 2019-nCov infection on the mucosa of oral cavity, bulk RNA-seq profiles from two public databases including  The Cancer <b>Genome</b> Atlas (TCGA) and Functional Annotation of The Mammalian Genome  Cap Analysis of Gene Expression (FANTOM5 CAGE) dataset were collected.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32053579">

Persons Evaluated for 2019 Novel Coronavirus - United States, January 2020.

</a>

<small>(PMID32053579</small>)

<br>...<b>Genetic</b> sequencing of isolates obtained from patients with pneumonia identified a novel coronavirus (2019-nCoV) as the etiology (1).

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31712093">

Rapid manipulation of the porcine epidemic diarrhea virus genome by CRISPR/Cas9 technology.

</a>

<small>(PMID31712093</small>)

<br>...In this study, a full-length infectious cDNA clone of the highly virulent PEDV <b>strain</b> AJ1102 was assembled in a bacterial artificial chromosome (BAC)...Importantly, it just took one week to construct the recombinant PEDV rAJ1102-DeltaORF3-EGFP using this method, providing a more efficient platform for PEDV <b>genome</b> manipulation, which could also be applied to other RNA viruses..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31979013">

Emerging Viruses without Borders: The Wuhan Coronavirus.

</a>

<small>(PMID31979013</small>)

<br>...We applaud the rapid release to the public of the <b>genome</b> sequence of the new virus by Chinese virologists, but we also believe that increased transparency on disease reporting and data sharing with international colleagues  are crucial for curbing the spread of this newly emerging virus to other parts of the world..

<td>Editorial</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...METHODS: Here we present a validated diagnostic workflow for  2019-nCoV, its design relying on close <b>genetic</b> relatedness of 2019-nCoV with SARS coronavirus, making use of synthetic nucleic acid technology.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31936476">

Isolation and Identification of Porcine Deltacoronavirus and Alteration of Immunoglobulin Transport Receptors in the Intestinal Mucosa of PDCoV-Infected Piglets.

</a>

<small>(PMID31936476</small>)

<br>...The <b>strain</b> CHN-JS-2017 was isolated  and identified by cytopathology, immunofluorescence assays, transmission electron microscopy, and sequence analysis...A nucleotide sequence alignment showed that the whole <b>genome</b> of CHN-JS-2017 is 97.4%-99.6% identical to other PDCoV strains... The pathogenicity of the CHN-JS-2017 <b>strain</b> was investigated in orally inoculated five-day-old piglets; the piglets developed acute, watery diarrhea, but all recovered and survived.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>... I consider the options of drug repurposing, developing neutralizing monoclonal antibody therapy, and an oligonucleotide strategy targeting the viral RNA <b>genome</b>, emphasizing the promise and pitfalls of these approaches.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31987001">

Genomic characterization of the 2019 novel human-pathogenic coronavirus isolated  from a patient with atypical pneumonia after visiting Wuhan.

</a>

<small>(PMID31987001</small>)

<br>...We performed bioinformatics analysis on a virus <b>genome</b> from a patient with 2019-nCoV infection and compared it with other related coronavirus genomes...Overall, the <b>genome</b> of 2019-nCoV has 89% nucleotide identity with bat SARS-like-CoVZXC21 and 82% with that of human SARS-CoV.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32020836">

RNA based mNGS approach identifies a novel human coronavirus from two individual  pneumonia cases in 2019 Wuhan outbreak.

</a>

<small>(PMID32020836</small>)

<br>...The entire viral <b>genome</b> is 29,881 nt in length (GenBank MN988668 and MN988669, Sequence Read Archive database Bioproject accession PRJNA601736) and is classified into beta-coronavirus genus...Phylogenetic analysis indicates that 2019-nCoV is close to coronaviruses (CoVs) circulating in Rhinolophus (Horseshoe bats), such as 98.7% nucleotide identity to partial RdRp gene of bat coronavirus strain BtCoV/4991 (GenBank KP876546, 370 nt sequence of RdRp and lack of other <b>genome</b> sequence) and 87.9% nucleotide identity to bat coronavirus strain bat-SL-CoVZC45  and bat-SL-CoVZXC21.

<td>Case Reports; Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32172672">

A tug-of-war between severe acute respiratory syndrome coronavirus 2 and host antiviral defence: lessons from other pathogenic viruses.

</a>

<small>(PMID32172672</small>)

<br>...We summarize current understanding of the induction of a proinflammatory cytokine storm by other highly pathogenic human coronaviruses, their <b>adaptation</b> to humans and their usurpation of the cell death programmes.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31859605">

Genetic manipulation of porcine deltacoronavirus reveals insights into NS6 and NS7 functions: a novel strategy for vaccine design.

</a>

<small>(PMID31859605</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767069">

Attenuation and characterization of porcine enteric alphacoronavirus strain GDS04 via serial cell passage.

</a>

<small>(PMID31767069</small>)

<br>...In this study, an original, highly virulent PEAV <b>strain</b> GDS04 was serially passaged  in Vero cells...Importantly, all P100-inoculated newborn piglets survived, indicating that P100 was an attenuated <b>variant</b>...Sequence analysis revealed that the virulent <b>strain</b> GDS04 had four, one, six and eleven amino acid differences in membrane, nucleocapsid, spike and ORF1ab proteins, respectively, from P100...Collectively, our research successfully prepared a PEAV attenuated <b>variant</b> which might serve as a live attenuated vaccine candidate against PEAV infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31852899">

SKP2 attenuates autophagy through Beclin1-ubiquitination and its inhibition reduces MERS-Coronavirus infection.

</a>

<small>(PMID31852899</small>)

<br>...<b>Genetic</b> or pharmacological inhibition of SKP2 decreases BECN1 ubiquitination, decreases BECN1 degradation and enhances  autophagic flux.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31463771">

Short hairpin RNAs targeting M and N genes reduce replication of porcine deltacoronavirus in ST cells.

</a>

<small>(PMID31463771</small>)

<br>...To study the potential of RNA interference (RNAi) as a strategy against PDCoV infection, two short hairpin RNA  (shRNA)-expressing plasmids (pGenesil-M and pGenesil-N) that targeted the M and N genes of PDCoV were constructed and transfected separately into swine testicular  (ST) cells, which were then infected with PDCoV <b>strain</b> HB-BD.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SA/SARS Virus">SARS Virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209163">

Serological and molecular findings during SARS-CoV-2 infection: the first case study in Finland, January to February 2020.

</a>

<small>(PMID32209163</small>)

<br>...The SARS-CoV-2/Finland/1/2020 virus strain was isolated, the <b>genome</b> showing a single  nucleotide substitution to the reference strain from Wuhan.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832902">

Virological and Epidemiological Situation in the Influenza Epidemic Seasons 2016/2017 and 2017/2018 in Poland.

</a>

<small>(PMID31832902</small>)

<br>...In this season, influenza B virus predominated, while the A/H1N1/pdm09 <b>strain</b> was most frequent among the influenza A subtypes.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31914458">

Preclinical evaluation of AT-527, a novel guanosine nucleotide prodrug with potent, pan-genotypic activity against hepatitis C virus.

</a>

<small>(PMID31914458</small>)

<br>...Despite the availability of highly effective direct-acting antiviral (DAA) regimens for the treatment of hepatitis C virus (HCV) infections, sustained viral response (SVR) rates remain suboptimal for difficult-to-treat patient populations such as those with HCV <b>genotype</b> 3, cirrhosis or prior treatment experience, warranting development of more potent HCV replication antivirals.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=6><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31982944">

A recombinant infectious bronchitis virus from a chicken with a spike gene closely related to that of a turkey coronavirus.

</a>

<small>(PMID31982944</small>)

<br>...Using viral metagenomics, the complete <b>genome</b> sequence of an infectious bronchitis virus (IBV) strain (named ahysx-1) from a fecal sample from a healthy  chicken in Anhui province, China, was determined...The <b>genome</b> sequence of ahysx-1  was found to be very similar to that of IBV strain ck/CH/LLN/131040 (KX252787), except for the spike gene region, which is similar to that of a turkey coronavirus strain (EU022526), suggesting that ahysx-1 is a recombinant...Further studies need to be performed to determine whether this recombinant IBV <b>strain</b> is pathogenic and whether it is transmitted between chickens and turkeys..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209118">

Using the spike protein feature to predict infection risk and monitor the evolutionary dynamic of coronavirus.

</a>

<small>(PMID32209118</small>)

<br>...The study may be beneficial for the surveillance of the <b>genome</b> mutation of coronavirus in the field..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31996413">

Discovery of Bat Coronaviruses through Surveillance and Probe Capture-Based Next-Generation Sequencing.

</a>

<small>(PMID31996413</small>)

<br>...Next-generation sequencing (NGS) is currently the preferred methodology for virus discovery to ensure unbiased sequencing of bat CoVs, considering their high <b>genetic</b> diversity...We discovered nine full genomes of bat CoVs in this study and revealed great <b>genetic</b> diversity for eight of them.IMPORTANCE Active surveillance is both urgent and essential to predict and mitigate the emergence of bat-origin CoV in humans and livestock...However, great <b>genetic</b> diversity increases the chance of homologous recombination among CoVs.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226286">

Zoonotic origins of human coronaviruses.

</a>

<small>(PMID32226286</small>)

<br>...Mutation and <b>adaptation</b> have driven the co-evolution of coronaviruses (CoVs) and  their hosts, including human beings, for thousands of years...Importantly, we compare and contrast the different HCoVs from a perspective of virus evolution and <b>genome</b> recombination.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31536759">

Comprehensive codon usage analysis of porcine deltacoronavirus.

</a>

<small>(PMID31536759</small>)

<br>...In this study, we analyzed the codon usage pattern of  the S gene using complete coding sequences and complete PDCoV genomes to gain a deeper understanding of their <b>genetic</b> relationships and evolutionary history...Our results revealed that different natural environments may have a significant impact on the <b>genetic</b> characteristics of the strains.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32083983">

Molecular characterization of Brazilian wild-type strains of bovine respiratory syncytial virus reveals genetic diversity and a putative new subgroup of the virus.

</a>

<small>(PMID32083983</small>)

<br>...For the BRSV G and F partial gene amplifications, RT-nested-PCR assays were performed with sequencing in both directions with forward and reverse primers used.Results: The G gene-based analysis revealed that two strains were highly similar to the BRSV sequences representative of subgroup III, including the Bayovac vaccine <b>strain</b>.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SW/Swine Diseases">Swine Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31805938">

High levels of unreported intraspecific diversity among RNA viruses in faeces of  neonatal piglets with diarrhoea.

</a>

<small>(PMID31805938</small>)

<br>...Next Generation Sequencing (NGS) deeply characterize the <b>genetic</b> diversity among rapidly mutating virus populations at the interspecific as well as the intraspecific level.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CH/Chiroptera">Chiroptera</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31473248">

Culture-dependent and metagenomic analysis of lesser horseshoe bats' gut microbiome revealing unique bacterial diversity and signatures of potential human pathogens.

</a>

<small>(PMID31473248</small>)

<br>...Therefore, high-throughput screening was used to understand the population structure, <b>genetic</b> diversity, and ecological role of the microorganisms.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/D/DO/Dog Diseases">Dog Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31745718">

Detection and molecular characterization of canine circovirus circulating in northeastern China during 2014-2016.

</a>

<small>(PMID31745718</small>)

<br>...Sequence analysis showed that there were two unique  amino acid changes in the Rep protein (N39S in the K1 <b>strain</b>, and T71A in the XF16 strain).

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31753544">

UV-C irradiation is able to inactivate pathogens found in commercially collected  porcine plasma as demonstrated by swine bioassay.

</a>

<small>(PMID31753544</small>)

<br>...Pigs negative for PCV-2 and PRRSV <b>genome</b> and antibodies were allotted to one of five groups (6 to 8 pigs/ group) and injected intra-peritoneally with 10 mL of their assigned inoculum at 50 d of age.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/H/HI/HIV-1">HIV-1</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32056509">

HIV-1 did not contribute to the 2019-nCoV genome.

</a>

<small>(PMID32056509</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/D/DE/Dengue Virus">Dengue Virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951602">

Validation of the easyscreen flavivirus dengue alphavirus detection kit based on  3base amplification technology and its application to the 2016/17 Vanuatu dengue  outbreak.

</a>

<small>(PMID31951602</small>)

<br>...The pan-alphavirus assay had a  sensitivity range of 10-25 copies per reaction depending on the viral <b>strain</b>.

<td>Journal Article; Validation Study</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RE/Respiratory Syncytial Virus Infections">Respiratory Syncytial Virus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31634614">

Epidemiological characteristics and phylogenic analysis of human respiratory syncytial virus in patients with respiratory infections during 2011-2016 in southern China.

</a>

<small>(PMID31634614</small>)

<br>...The prevalent RSV-A <b>genotype</b> in 2011-2012 was NA1, close to Chongqing and Brazil, but a new Hong Kong ON1 genotype was introduced and became the prevalent genotype in Guangzhou in 2014-2015.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31878875">

The ability of single genes vs full genomes to resolve time and space in outbreak analysis.

</a>

<small>(PMID31878875</small>)

<br>...BACKGROUND: Inexpensive pathogen <b>genome</b> sequencing has had a transformative effect on the field of phylodynamics, where ever increasing volumes of data have  promised real-time insight into outbreaks of infectious disease...Despite its utility, whole <b>genome</b> sequencing of pathogens has not been adopted universally and targeted sequencing of loci is common in some pathogen-specific fields.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/M/MA/Macrophages">Macrophages</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710162">

Tumor-associated macrophages secrete CC-chemokine ligand 2 and induce tamoxifen resistance by activating PI3K/Akt/mTOR in breast cancer.

</a>

<small>(PMID31710162</small>)

<br>...Herein, we aimed to elucidate the relationship between TAM  and the endocrine-resistant <b>phenotype</b> of breast cancer.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RE/Real-Time Polymerase Chain Reaction">Real-Time Polymerase Chain Reaction</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31884173">

A new multiplex RT-qPCR method for the simultaneous detection and discrimination  of Zika and chikungunya viruses.

</a>

<small>(PMID31884173</small>)

<br>...METHODS: Two methods targeting different <b>genome</b> segments were selected from the literature for each virus...These were adapted for high <b>genome</b> coverage and combined in a four-plex assay that was thoroughly validated in-house.

<td>Journal Article; Validation Study</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/R/RN/RNA Replicase">RNA Replicase</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...The enterovirus 71 (EV71) 3Dpol is an RNA-dependent RNA polymerase (RdRP) that plays the central role in the viral <b>genome</b> replication, and is an important target in antiviral studies.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Camelus">Camelus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31532019">

Old World camels in a modern world - a balancing act between conservation and genetic improvement.

</a>

<small>(PMID31532019</small>)

<br>...We cannot emphasise enough the importance of balancing the need for improving camel production traits with maintaining the genetic diversity in two domestic species with specific physiological <b>adaptation</b> to a desert environment..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CA/Cattle">Cattle</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31677831">

Associations between maternal characteristics and health, survival, and performance of dairy heifers from birth through first lactation.

</a>

<small>(PMID31677831</small>)

<br>...41 kg), had greater <b>genetic</b> merit for production traits (e.g., genomic estimated breeding value for milk yield: 875 vs...37%), and had reduced performance in the first lactation when considering their <b>genetic</b> merit (e.g., 305-d yield of energy-corrected milk: 11,270 vs...25%) even though <b>genetic</b> merit for production traits were similar  (e.g., genomic estimated breeding value for milk: 744 vs.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/H/HE/Hepatitis E virus">Hepatitis E virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31758397">

Methyltransferase of a cell culture-adapted hepatitis E inhibits the MDA5 receptor signaling pathway.

</a>

<small>(PMID31758397</small>)

<br>...As a mouse model is not available, a recent development of a cell culture-adapted HEV <b>strain</b> (47832c) is  considered as a very important tools for molecular analysis of HEV pathogenesis in cells...Previously, we demonstrated that HEV-encoded methyltransferase (MeT) encoded by the 47832c <b>strain</b> inhibits MDA5- and RIG-I-mediated activation of interferon beta (IFN-beta) promoter...A deeper understanding of MeTmediated  suppression of IFN-beta expression would provide basis of the cell culture <b>adaptation</b> of HEV..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/S/SP/Spike Glycoprotein, Coronavirus">Spike Glycoprotein, Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32057769">

The spike glycoprotein of the new coronavirus 2019-nCoV contains a furin-like cleavage site absent in CoV of the same clade.

</a>

<small>(PMID32057769</small>)

<br>...Its <b>genome</b> has been sequenced and the genomic information promptly released...Despite a high similarity with the <b>genome</b> sequence of SARS-CoV and SARS-like CoVs, we identified a peculiar furin-like cleavage site in the Spike protein of the 2019-nCoV, lacking in the other SARS-like CoVs.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EN/Endothelial Cells">Endothelial Cells</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32187326">

Involvement of lipid microdomains in human endothelial cells infected by Streptococcus agalactiae type III belonging to the hypervirulent ST-17.

</a>

<small>(PMID32187326</small>)

<br>...agalactiae  <b>strain</b> to HUVEC cells...agalactiae <b>strain</b> in HUVEC cells.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EX/Extracellular Vesicles">Extracellular Vesicles</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31733447">

Deiminated proteins in extracellular vesicles and serum of llama (Lama glama)-Novel insights into camelid immunity.

</a>

<small>(PMID31733447</small>)

<br>...EVs are found in most body fluids and participate in cellular communication via transfer of cargo proteins and <b>genetic</b>  material.

<td>Journal Article; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31813465">

Evolutionary Medicine of Retroviruses in the Human Genome.

</a>

<small>(PMID31813465</small>)

<br>...However, certain viruses have entered the human <b>genome</b>...Of the human <b>genome</b>, approximately 45% is composed of transposable elements (long interspersed nuclear elements [LINEs], short interspersed nuclear  elements [SINEs] and transposons) and 5-8% is derived from viral sequences with similarity to infectious retroviruses...Accumulation of viral sequences has created the current human <b>genome</b>...Second, we review endogenous retroviruses in the human <b>genome</b> and diseases associated with endogenous retroviruses.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32004758">

Full-genome evolutionary analysis of the novel corona virus (2019-nCoV) rejects the hypothesis of emergence as a result of a recent recombination event.

</a>

<small>(PMID32004758</small>)

<br>...Our objectives were to characterize the <b>genetic</b> relationships  of the 2019-nCoV and to search for putative recombination within the subgenus of  sarbecovirus...RESULTS: Our analysis suggests that the 2019-nCoV although closely related to BatCoV RaTG13 sequence throughout the <b>genome</b> (sequence similarity 96.3%), shows discordant clustering with the Bat_SARS-like coronavirus sequences...CONCLUSIONS: The levels of genetic similarity between the 2019-nCoV and RaTG13 suggest that the latter does not provide the exact <b>variant</b> that caused the outbreak in humans, but the hypothesis that 2019-nCoV has originated from bats is very likely...We show evidence that the novel coronavirus  (2019-nCov) is not-mosaic consisting in almost half of its <b>genome</b> of a distinct lineage within the betacoronavirus.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32127124">

Early transmission patterns of coronavirus disease 2019 (COVID-19) in travellers  from Wuhan to Thailand, January 2020.

</a>

<small>(PMID32127124</small>)

<br>...Both were independent introductions on separate flights, discovered with thermoscanners and confirmed with RT-PCR and <b>genome</b> sequencing.

<td>Case Reports; Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/O/OR/Oropharynx">Oropharynx</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32080990">

Virus Isolation from the First Patient with SARS-CoV-2 in Korea.

</a>

<small>(PMID32080990</small>)

<br>...Phylogenetic analyses of whole <b>genome</b> sequences showed that it clustered with other SARS-CoV-2 reported from Wuhan..

<td>Case Reports; Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=variant+OR+adaptation+OR+phenotype+OR+genotype+OR+genetic+OR+genome+OR+strain&from=CORD19#/C/CE/Cell Line">Cell Line</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31683198">

Characterisation of Crandell-Rees Feline Kidney (CRFK) cells as mesenchymal in phenotype.

</a>

<small>(PMID31683198</small>)

<br>...Confusion exists as to whether CRFK are epithelial or mesenchymal in <b>phenotype</b>.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 6751 matches before 2019/12

<hr><a name="task1k"><b>Task1k Kaggle Prompt:</b> Immune response and immunity</a><p><b>Results:</b><p>

Searching for (immune OR immunity OR immunoglobulin) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/I/IM/Immunity, Innate">Immunity, Innate</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/V/VI/Viral Nonstructural Proteins">Viral Nonstructural Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/H/HO/Host-Pathogen Interactions">Host-Pathogen Interactions</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/A/AD/Adjuvants, Immunologic">Adjuvants, Immunologic</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CH/Chiroptera">Chiroptera</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/I/IM/Immunologic Factors">Immunologic Factors</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CO/Collectins">Collectins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=9><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32208840">

Clinical observation and management of COVID-19 patients.

</a>

<small>(PMID32208840</small>)

<br>...Furthermore, he observes the significant abnormality of coagulation function and proposes that the early intravenous <b>immunoglobulin</b> and low molecular weight heparin anticoagulation therapy are very  important.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32092539">

Is COVID-19 receiving ADE from other coronaviruses?

</a>

<small>(PMID32092539</small>)

<br>...ADE modulates the <b>immune</b> response and can elicit sustained inflammation, lymphopenia, and/or cytokine storm, one or all of which have been documented in severe cases and deaths.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32113846">

In silico screening of Chinese herbal medicines with the potential to directly inhibit 2019 novel coronavirus.

</a>

<small>(PMID32113846</small>)

<br>...Network pharmacology analysis predicted that the general  in vivo roles of these 26 herbal plants were related to regulating viral infection, <b>immune</b>/inflammation reactions and hypoxia response.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31950289">

Recombinant adenovirus carrying a core neutralizing epitope of porcine epidemic diarrhea virus and heat-labile enterotoxin B of Escherichia coli as a mucosal vaccine.

</a>

<small>(PMID31950289</small>)

<br>...Three intramuscular or oral vaccinations with rAd-LTB-COE at two-week intervals induced robust humoral and mucosal <b>immune</b> responses...Moreover, a cell-mediated <b>immune</b> response was promoted  in immunized mice, and the neutralizing antibody inhibited both the vaccine strain and the emerging PEDV isolate...Immunization experiments in piglets revealed that rAd-LTB-COE was immunogenic and induced good <b>immune</b> responses in piglets.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040667">

Critical care management of adults with community-acquired severe respiratory viral infection.

</a>

<small>(PMID32040667</small>)

<br>...Several adjunctive pharmacologic interventions have been studied for their immunomodulatory effects, including macrolides, corticosteroids, cyclooxygenase-2 inhibitors, sirolimus, statins, anti-influenza  <b>immune</b> plasma, and vitamin C, but none is recommended at present in severe RVIs.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31936476">

Isolation and Identification of Porcine Deltacoronavirus and Alteration of Immunoglobulin Transport Receptors in the Intestinal Mucosa of PDCoV-Infected Piglets.

</a>

<small>(PMID31936476</small>)

<br>...The neonatal Fc receptor (FcRn) and polymeric <b>immunoglobulin</b> receptor (pIgR) are crucial immunoglobulin (Ig) receptors for the  transcytosis ofimmunoglobulin G (IgG), IgA, or IgM.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...The proposal is a biologic that blocks 2019-nCoV entry using a soluble version of the viral receptor, angiotensin-converting enzyme 2 (ACE2), fused to an immunoglobulin Fc domain, providing a neutralizing antibody with maximal breath to avoid any viral escape,  while also helping to recruit the <b>immune</b> system to build lasting immunity.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226289">

Perspectives on therapeutic neutralizing antibodies against the Novel Coronavirus SARS-CoV-2.

</a>

<small>(PMID32226289</small>)

<br>...Herein, the host <b>immune</b> responses against SARS-CoV discussed in this review provide implications for developing NAbs and understanding clinical interventions against SARS-CoV-2.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32221306">

Characterization of spike glycoprotein of SARS-CoV-2 on virus entry and its immune cross-reactivity with SARS-CoV.

</a>

<small>(PMID32221306</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/I/IM/Immunity, Innate">Immunity, Innate</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513433">

Antiviral immunity is impaired in COPD patients with frequent exacerbations.

</a>

<small>(PMID31513433</small>)

<br>...The aim of this study was to determine the innate <b>immune</b> mechanisms that underlie susceptibility to frequent exacerbations in COPD...We measured sputum expression of <b>immune</b> mediators and bacterial loads in samples from patients with COPD at stable state and during virus-associated exacerbations...In vitro <b>immune</b> responses to rhinovirus infection in differentiated primary bronchial epithelial cells (BECs) sampled from patients with COPD were additionally evaluated...Frequent exacerbators had reduced sputum cell mRNA expression of the antiviral <b>immune</b> mediators type I and III interferons and reduced interferon-stimulated gene (ISG) expression when clinically stable and during virus-associated exacerbation...A role for epithelial cell-intrinsic innate <b>immune</b> dysregulation was identified: induction of interferons and ISGs during in  vitro rhinovirus (RV) infection was also impaired in differentiated BECs from frequent exacerbators...These data implicate deficient airway innate <b>immunity</b> involving epithelial cells in the increased propensity to exacerbations observed in some patients with COPD...Therapeutic approaches to boost innate antimicrobial <b>immunity</b> in the lung could be a viable strategy for prevention and treatment of frequent exacerbations..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/V/VI/Viral Nonstructural Proteins">Viral Nonstructural Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31359346">

The DEAD-Box RNA Helicase DDX1 Interacts with the Viral Protein 3D and Inhibits Foot-and-Mouth Disease Virus Replication.

</a>

<small>(PMID31359346</small>)

<br>...DDX1 was reported to either inhibit or facilitate viral replication and regulate host innate <b>immune</b> responses.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/H/HO/Host-Pathogen Interactions">Host-Pathogen Interactions</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152948">

Siglecs at the Host-Pathogen Interface.

</a>

<small>(PMID32152948</small>)

<br>...The subset known as CD33-related Siglecs is principally inhibitory receptors that suppress leukocyte activation, and recent research has shown that  a number of bacterial pathogens use Sia mimicry to engage these Siglecs as an <b>immune</b> evasion strategy...Conversely, Siglec-1 is a macrophage phagocytic receptor that engages GBS and other sialylated bacteria to promote effective phagocytosis  and antigen presentation for the adaptive <b>immune</b> response, whereas certain viruses and parasites use Siglec-1 to gain entry to immune cells as a proximal step in the infectious process...Siglecs are positioned in crosstalk with other host innate <b>immune</b> sensing pathways to modulate the immune response to infection  in complex ways.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/A/AD/Adjuvants, Immunologic">Adjuvants, Immunologic</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31856726">

Prediction of novel mouse TLR9 agonists using a random forest approach.

</a>

<small>(PMID31856726</small>)

<br>...BACKGROUND: Toll-like receptor 9 is a key innate <b>immune</b> receptor involved in detecting infectious diseases and cancer...TLR9 activates the innate <b>immune</b> system following the recognition of single-stranded DNA oligonucleotides (ODN) containing unmethylated cytosine-guanine (CpG) motifs.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CH/Chiroptera">Chiroptera</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31910439">

Mating strategy is determinant of adenovirus prevalence in European bats.

</a>

<small>(PMID31910439</small>)

<br>...Alternatively, bat species with more promiscuous behavior may develop  a stronger <b>immune</b> system.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513894">

Production of anti-Trichophyton rubrum egg yolk immunoglobulin and its therapeutic potential for treating dermatophytosis.

</a>

<small>(PMID31513894</small>)

<br>...The aim of this study was to estimate the therapeutic potential of specific egg yolk <b>immunoglobulin</b> (IgY) on dermatophytosis caused by Trichophyton rubrum.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/I/IM/Immunologic Factors">Immunologic Factors</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542433">

Structure-function and application of plant lectins in disease biology and immunity.

</a>

<small>(PMID31542433</small>)

<br>...We found that many plant lectins mediate its microbicidal activity by triggering  host <b>immune</b> responses that result in the release of several cytokines followed by activation of effector mechanism.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31813465">

Evolutionary Medicine of Retroviruses in the Human Genome.

</a>

<small>(PMID31813465</small>)

<br>...Humans are infected with many viruses, and the <b>immune</b> system mostly removes viruses and the infected cells.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CO/Collectins">Collectins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152944">

Collectins: Innate Immune Pattern Recognition Molecules.

</a>

<small>(PMID32152944</small>)

<br>...Collectins are collagen-containing C-type (calcium-dependent) lectins which are important pathogen pattern recognising innate <b>immune</b> molecules...Collectins can also potentiate the adaptive <b>immune</b> response via antigen presenting cells such as macrophages and dendritic cells through modulation of cytokines and chemokines, thus they can act as a link between innate and adaptive immunity.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=immune+OR+immunity+OR+immunoglobulin&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31733447">

Deiminated proteins in extracellular vesicles and serum of llama (Lama glama)-Novel insights into camelid immunity.

</a>

<small>(PMID31733447</small>)

<br>...In serum, 103 deiminated proteins were overall identified,  including key <b>immune</b> and metabolic mediators including complement components, immunoglobulin-based nanobodies, adiponectin and heat shock proteins...This is the first report  of deiminated proteins in serum and EVs of a camelid species, highlighting a hitherto unrecognized post-translational modification in key <b>immune</b> and metabolic proteins in camelids, which may be translatable to and inform a range of human metabolic and inflammatory pathologies..

<td>Journal Article; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 5060 matches before 2019/12

<hr><a name="task1l"><b>Task1l Kaggle Prompt:</b> Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings</a><p><b>Results:</b><p>

Searching for (staff OR cross OR safety) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/I/IN/Infection Control">Infection Control</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/V/VI/Viruses">Viruses</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/D/DE/Dengue Virus">Dengue Virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/P/PL/Plasma">Plasma</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31823763">

Evaluation of the influenza sentinel surveillance system in the Democratic Republic of Congo, 2012-2015.

</a>

<small>(PMID31823763</small>)

<br>...It was reported that the ISSS contributed to: (i) a better understanding of the epidemiology, circulating patterns and proportional contribution of influenza virus among patients with ILI or SARI; (ii) acquisition of new key competences related to influenza surveillance and diagnosis; and (iii) continuous education of surveillance <b>staff</b> and clinicians at sentinel sites about influenza.

<td>Evaluation Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32092296">

Coronavirus (COVID-19) Outbreak: What the Department of Radiology Should Know.

</a>

<small>(PMID32092296</small>)

<br>...Moreover, the authors review precautions and <b>safety</b> measures for radiology department personnel to manage patients with known or suspected NCIP...Implementation of a robust plan in the radiology department is required to prevent further transmission of the virus to  patients and department <b>staff</b> members..

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183934">

Post-discharge surveillance and positive virus detection in two medical staff recovered from coronavirus disease 2019 (COVID-19), China, January to February 2020.

</a>

<small>(PMID32183934</small>)

<br>...Since December 2019, 62 medical <b>staff</b> of Zhongnan Hospital in Wuhan, China have been hospitalised with coronavirus disease 2019.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32193904">

Drive-Through Screening Center for COVID-19: a Safe and Efficient Screening System against Massive Community Outbreak.

</a>

<small>(PMID32193904</small>)

<br>...Increased testing capacity over 100 tests per day and prevention of cross-infection between testees in the waiting space are the major  advantages, while protection of <b>staff</b> from the outdoor atmosphere is challenging.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/I/IN/Infection Control">Infection Control</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32105633">

Staff safety during emergency airway management for COVID-19 in Hong Kong.

</a>

<small>(PMID32105633</small>)

<br>....

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...Innovating built facility to safely treat patients with highly pathogenic infectious diseases is urgently need, not only to prevent the spread of infection from patients to healthcare workers but also to offer provision of relatively invasive organ support, whenever considered appropriate, without posing additional risk to <b>staff</b>.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/V/VI/Viruses">Viruses</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31881820">

IILLS: predicting virus-receptor interactions based on similarity and semi-supervised learning.

</a>

<small>(PMID31881820</small>)

<br>...The 10-fold <b>cross</b> validation (10CV) and leave one out cross validation (LOOCV) are used to assess the prediction performance of our method...CONLUSION: The experiment results show that IILLS achieves the AUC values of 0.8675 and 0.9061 with the 10-fold <b>cross</b> validation and leave-one-out cross validation (LOOCV), respectively, which illustrates that  IILLS is superior to the competing methods.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209118">

Using the spike protein feature to predict infection risk and monitor the evolutionary dynamic of coronavirus.

</a>

<small>(PMID32209118</small>)

<br>...BACKGROUND: Coronavirus can <b>cross</b> the species barrier and infect humans with a severe respiratory syndrome.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/D/DE/Dengue Virus">Dengue Virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951602">

Validation of the easyscreen flavivirus dengue alphavirus detection kit based on  3base amplification technology and its application to the 2016/17 Vanuatu dengue  outbreak.

</a>

<small>(PMID31951602</small>)

<br>...No <b>cross</b> reactivity was observed with a number of commonly encountered viral strains.

<td>Journal Article; Validation Study</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/P/PL/Plasma">Plasma</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31753544">

UV-C irradiation is able to inactivate pathogens found in commercially collected  porcine plasma as demonstrated by swine bioassay.

</a>

<small>(PMID31753544</small>)

<br>...In previous studies we found that the application of ultraviolet light C (UV-C) in liquid plasma that was inoculated with a variety of bacteria or viruses of importance in the swine industry can be considered as redundant <b>safety</b> steps because in general achieve around 4 logs reduction for most of these pathogens...However, the final validation of the UV-C light as <b>safety</b> feature should be conducted with commercial liquid plasma and using the pig bioassay model.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=staff+OR+cross+OR+safety&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31733447">

Deiminated proteins in extracellular vesicles and serum of llama (Lama glama)-Novel insights into camelid immunity.

</a>

<small>(PMID31733447</small>)

<br>...PAD homologues were identified in llama serum by Western blotting, via <b>cross</b> reaction with human PAD antibodies, and detected at an expected 70kDa size.

<td>Journal Article; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 1373 matches before 2019/12

<hr><a name="task1m"><b>Task1m Kaggle Prompt:</b> Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings</a><p><b>Results:</b><p>

Searching for (PPE or protective) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/E/EP/Epidemics">Epidemics</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=6><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240128">

Asymptomatic and Presymptomatic SARS-CoV-2 Infections in Residents of a Long-Term Care Skilled Nursing Facility - King County, Washington, March 2020.

</a>

<small>(PMID32240128</small>)

<br>...Once a confirmed case is identified in an SNF, all residents should be placed on isolation precautions if possible (3), with considerations for extended use or reuse of personal protective equipment (<b>PPE</b>) as needed (4)..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support preparedness in low and middle income countries  especially, take strong global interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal <b>protective</b> tools such as facemask to general population, and quickly initiate research projects on drug and vaccine development.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040190">

A qualitative study of zoonotic risk factors among rural communities in southern  China.

</a>

<small>(PMID32040190</small>)

<br>...Data were collected through ethnographic interviews and field observations, and thematically coded and analysed to identify both risk and <b>protective</b> factors for zoonotic disease emergence at the individual, community and policy levels.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Ultimately, the outbreak could be controlled with a <b>protective</b> vaccine to prevent 2019-nCoV infection...Such a treatment could help infected patients before a <b>protective</b> vaccine is developed and widely available in the coming months to year(s)..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226293">

The COVID-19 outbreak and psychiatric hospitals in China: managing challenges through mental health service reform.

</a>

<small>(PMID32226293</small>)

<br>...Possible reasons quoted in the report were the lack of caution regarding the COVID-19 outbreak in  January and insufficient supplies of <b>protective</b> gear.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31526954">

Detection of MERS-CoV antigen on formalin-fixed paraffin-embedded nasal tissue of alpacas by immunohistochemistry using human monoclonal antibodies directed against different epitopes of the spike protein.

</a>

<small>(PMID31526954</small>)

<br>...In recent years, several investigators developed <b>protective</b> antibodies which could be used  as prophylaxis in prospective human epidemics.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=PPE+or+protective&from=CORD19#/E/EP/Epidemics">Epidemics</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31905206">

Risk perception and behavioral change during epidemics: Comparing models of individual and collective learning.

</a>

<small>(PMID31905206</small>)

<br>...It requires a deep understanding of how individuals perceive risks and communicate about the effectiveness of <b>protective</b>  measures, highlighting learning and social interaction as the core mechanisms driving such processes...However, little attention has been paid to the role of intelligent learning in risk appraisal and <b>protective</b> decisions, whether used in an individual or a collective process.

<td>Comparative Study; Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

</table>

<p>There are also 1359 matches before 2019/12

<hr><a name="task1n"><b>Task1n Kaggle Prompt:</b> Role of the environment in transmission</a><p><b>Results:</b><p>

Searching for (tried: humidity transmission, temperature transmission, environment transmission.) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



"""



h = display(HTML(htmlresults))

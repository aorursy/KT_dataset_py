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
htmlprompt="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<h2>CORD-19 Task Details</h2>

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=587">What do we know about non-pharmaceutical interventions?</A>



<p><strong>What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?</strong></p>





<table class=l border=1><tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.<td>infrastructure OR scale NPI<td>Task6a results below

<tr><td>Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.<td>distancing OR cleaning OR wash OR washing OR cover OR mask OR touch OR quarantine<td>Task6b results below

<tr><td>Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.<td>distancing OR closure OR ban OR lockdown OR lock-down OR quarantine<td>Task6c results below

<tr><td>Methods to control the spread in communities, barriers to compliance and how these vary among different populations..<td>community compliance<td>Task6d results below

<tr><td>Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.<td>cost model<td>Task6e results below

<tr><td>Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.<td>policy AND compliance<td>Task6f results below

<tr><td>Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.<td>economic impact<td>Task6g results below

</table>

"""



h = display(HTML(htmlprompt))



htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<hr><a name="task6a"><b>Task6a Kaggle Prompt:</b> Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.</a><p><b>Results:</b><p>

Searching for (infrastructure OR scale NPI) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=infrastructure+OR+scale+NPI&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=infrastructure+OR+scale+NPI&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=infrastructure+OR+scale+NPI&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...Control material is made available through European Virus Archive - Global (EVAg), a European Union <b>infrastructure</b> project.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=infrastructure+OR+scale+NPI&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...Preparedness including management of complex humanitarian crises with community distrust is a  cornerstone in response to high consequence emerging infectious disease outbreaks and imposes strengthening of the public health response <b>infrastructure</b> and emergency outbreak systems in high-risk regions..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 167 matches before 2019/12

<hr><a name="task6b"><b>Task6b Kaggle Prompt:</b> Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.</a><p><b>Results:</b><p>

Searching for (distancing OR cleaning OR wash OR washing OR cover OR mask OR touch OR quarantine) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/C/CO/Coptis">Coptis</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=10><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...The measures to prevent transmission was very successful at early  stage, the next steps on the COVID-19 infection should be focused on early isolation of patients and <b>quarantine</b> for close contacts in families and communities in Beijing.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238336">

Global Telemedicine Implementation and Integration Within Health Systems to Fight the COVID-19 Pandemic: A Call to Action.

</a>

<small>(PMID32238336</small>)

<br>...The response strategy included early diagnosis, patient isolation, symptomatic monitoring of contacts as well as suspected and confirmed cases, and public health <b>quarantine</b>.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217507">

Assessment of Health Information About COVID-19 Prevention on the Internet: Infodemiological Study.

</a>

<small>(PMID32217507</small>)

<br>...The most mentioned WHO preventive measure was "<b>wash</b> your hands frequently" (n=65, 81%)...The analysis by type of author (official public health  organizations versus digital media) revealed significant differences regarding the recommendation to wear a <b>mask</b> when you are healthy only if caring for a person with suspected COVID-19 (odds ratio [OR] 4.39)...According to the country of publication (Spain versus the United States), significant differences were detected regarding some recommendations such as "<b>wash</b> your hands frequently" (OR  9.82), "cover your mouth and nose with your bent elbow or tissue when you cough or sneeze" (OR 4.59), or "stay home if you feel unwell" (OR 0.31).

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32052841">

Isolation, quarantine, social distancing and community containment: pivotal role  for old-style public health measures in the novel coronavirus (2019-nCoV) outbreak.

</a>

<small>(PMID32052841</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183930">

Estimating the asymptomatic proportion of coronavirus disease 2019 (COVID-19) cases on board the Diamond Princess cruise ship, Yokohama, Japan, 2020.

</a>

<small>(PMID32183930</small>)

<br>...On 5 February 2020, in Yokohama, Japan, a cruise ship hosting 3,711 people underwent a 2-week <b>quarantine</b> after a former passenger was found with COVID-19 post-disembarking...Most infections occurred before the <b>quarantine</b> start..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>...Specific up-to-date and accurate health information (e.g., treatment, local outbreak situation) and particular precautionary measures (e.g., hand hygiene, wearing a <b>mask</b>) were associated with  a lower psychological impact of the outbreak and lower levels of stress, anxiety, and depression (p < 0.05).

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Current efforts are focused on containment and <b>quarantine</b> of infected individuals.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226295">

COVID-19: what has been learned and to be learned about the novel coronavirus disease.

</a>

<small>(PMID32226295</small>)

<br>...We will <b>cover</b> the basics about the epidemiology, etiology, virology, diagnosis, treatment, prognosis, and prevention of the disease.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231374">

Data-based analysis, modelling and forecasting of the COVID-19 outbreak.

</a>

<small>(PMID32231374</small>)

<br>...<b>quarantine</b> and hospitalization of infected individuals), but mainly because of the fact that the actual cumulative numbers of infected and recovered cases in the population most likely are much higher than the reported ones.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/C/CO/Coptis">Coptis</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/30963783">

Coptidis Rhizoma: a comprehensive review of its traditional uses, botany, phytochemistry, pharmacology and toxicology.

</a>

<small>(PMID30963783</small>)

<br>...The extracts/compounds isolated from CR <b>cover</b> a wide pharmacological spectrum, including antibacterial, antivirus, antifungal, antidiabetic, anticancer and cardioprotective effects.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+cleaning+OR+wash+OR+washing+OR+cover+OR+mask+OR+touch+OR+quarantine&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046819">

Incubation period of 2019 novel coronavirus (2019-nCoV) infections among travellers from Wuhan, China, 20-28 January 2020.

</a>

<small>(PMID32046819</small>)

<br>...These values should help inform 2019-nCoV case definitions and appropriate <b>quarantine</b> durations..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 672 matches before 2019/12

<hr><a name="task6c"><b>Task6c Kaggle Prompt:</b> Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.</a><p><b>Results:</b><p>

Searching for (distancing OR closure OR ban OR lockdown OR lock-down OR quarantine) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+closure+OR+ban+OR+lockdown+OR+lock-down+OR+quarantine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+closure+OR+ban+OR+lockdown+OR+lock-down+OR+quarantine&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=7><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+closure+OR+ban+OR+lockdown+OR+lock-down+OR+quarantine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...The measures to prevent transmission was very successful at early  stage, the next steps on the COVID-19 infection should be focused on early isolation of patients and <b>quarantine</b> for close contacts in families and communities in Beijing.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238336">

Global Telemedicine Implementation and Integration Within Health Systems to Fight the COVID-19 Pandemic: A Call to Action.

</a>

<small>(PMID32238336</small>)

<br>...The response strategy included early diagnosis, patient isolation, symptomatic monitoring of contacts as well as suspected and confirmed cases, and public health <b>quarantine</b>.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32052841">

Isolation, quarantine, social distancing and community containment: pivotal role  for old-style public health measures in the novel coronavirus (2019-nCoV) outbreak.

</a>

<small>(PMID32052841</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183930">

Estimating the asymptomatic proportion of coronavirus disease 2019 (COVID-19) cases on board the Diamond Princess cruise ship, Yokohama, Japan, 2020.

</a>

<small>(PMID32183930</small>)

<br>...On 5 February 2020, in Yokohama, Japan, a cruise ship hosting 3,711 people underwent a 2-week <b>quarantine</b> after a former passenger was found with COVID-19 post-disembarking...Most infections occurred before the <b>quarantine</b> start..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Current efforts are focused on containment and <b>quarantine</b> of infected individuals.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231374">

Data-based analysis, modelling and forecasting of the COVID-19 outbreak.

</a>

<small>(PMID32231374</small>)

<br>...<b>quarantine</b> and hospitalization of infected individuals), but mainly because of the fact that the actual cumulative numbers of infected and recovered cases in the population most likely are much higher than the reported ones.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=distancing+OR+closure+OR+ban+OR+lockdown+OR+lock-down+OR+quarantine&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046819">

Incubation period of 2019 novel coronavirus (2019-nCoV) infections among travellers from Wuhan, China, 20-28 January 2020.

</a>

<small>(PMID32046819</small>)

<br>...These values should help inform 2019-nCoV case definitions and appropriate <b>quarantine</b> durations..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 288 matches before 2019/12

<hr><a name="task6d"><b>Task6d Kaggle Prompt:</b> Methods to control the spread in communities, barriers to compliance and how these vary among different populations..</a><p><b>Results:</b><p>

Searching for (community compliance) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 13 matches before 2019/12

<hr><a name="task6e"><b>Task6e Kaggle Prompt:</b> Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.</a><p><b>Results:</b><p>

Searching for (cost model) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 78 matches before 2019/12

<hr><a name="task6f"><b>Task6f Kaggle Prompt:</b> Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.</a><p><b>Results:</b><p>

Searching for (policy AND compliance) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 13 matches before 2019/12

<hr><a name="task6g"><b>Task6g Kaggle Prompt:</b> Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.</a><p><b>Results:</b><p>

Searching for (economic impact) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 128 matches before 2019/12

"""



h = display(HTML(htmlresults))

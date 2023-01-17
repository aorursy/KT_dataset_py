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
htmlpros="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<p>

<h2>Pros and cons of my approach</h2>

<table class=l border=1>

<tr><th>Risk</th><th>Example Failure</th><th>Design</th>

<tr><td>	1.1 Non-trustworthy sources<td>Unscientific or fictitious source<td>Use CORD-19 dataset.<br>Pro: which are primarily sources from a PUBMED search of documents that were

written and published for professionals (in medicine, government, etc.). They are not all peer-reviewed (not a panacaea anyway) but they are expected to be scientifically trustworthy.<br>Con: CORD-19 are "open" sources. Better sources may be excluded.

<tr><td rowspan=4>1.2 Inappropriate use of trustworthy sources<td rowspan=4>Result out of context. (E.g. incubation period from a paper about mice, or SARS, which is a different virus)

	<td>Use results from titles and abstracts, not full text. <br>Pro: Full-text search is likely to find many documents with the terms that were intended to be mentioned only in passing, and in a different context, unnecessarily multiplying the quantity, but not the quality of results.<br>Pro: Titles (and most abstracts) are most often written from a "low-context" perspective using wording that is precise and most carefully considered. This is in contrast to sentences from full text which are very high-context. We avoid excerpting high-context sentences for presentation in search results so that we avoid the risk of misinterpretation due to taking them out of the author's context and presenting them in a very different context.<br>Con: Important findings may not be described in the abstract using search terms.

<tr><td>Avoid use of secondary MeSH headings.<br>Pro: A source with trustworthy conclusions should be categorized under a primary MeSH heading. Secondary headings are inconsistently applied and using them decreases reliability.<br>Con: Secondary headings, if they are accurate, could be useful in limiting search results further than just primary Headings. (Due to the inconsistent application of secondary headings, there are better ways of doing this than secondary headings, though. Using the alternative primary heading is better, for example.)

<tr><td>Use Human-in-the-loop to show context and interactively limit sources (e.g. by MeSH header, date range, paper type.)

<br>Pro: Human can apply and review individual criteria for their circumstances

<br>Con: "Cluttered" user interface that requires some experience to become familiar and use well. 

<tr><td>Date range limit and sort by date range descending. [See note 1 below.]

<tr><td rowspan=7>1.3 Missed sources<td rowspan=4>Search terms too narrow, lacking synonyms

	<td>Show search results within MeSH subject context.

	<br>Pro: Sources with similar topics are grouped into a category. Finding one document finds the category, and that means other articles that are similar. Synonym terms will be in those documents.

	<br>Con: MeSH subjects are too broad.

<tr><td>Use synonyms in search terms Caveat: (Currently human-directed, as was done in CORD-19 query)

	<br>Pro: avoids creating a good synonym dictionary

	<br>Con: Extra work and likely to lead non-experts to searches that are too narrow.

<tr><td>Show links to nearby categories that were used, because MeSH tagging for articles is not 100% reliable.

	<br>Pro: Tolerates the categorization imprecision that exists in any subject index, by showing "adjacent" categories, parents, and children.

	<br>Con: MeSH trees are not without flaws, and may be unfamiliar to users. Makes searching harder because several pages need to be consulted, not just one.

<tr><td>Show articles at multiple MeSH tree locations, not just one.

	<br>Pro: Articles are multi-faceted and should not go into a subject tree at one heading.

	<br>Con: Articles will be seen more than once when searching several subject pages.

<tr><td rowspan=2>Sources outside CORD-19<td>Show concept keywords for making external searches

	<br>Pro: Allows consulting other systems

	<br>Con: Would be better to integrate the other sources into one system.

<tr><td>Deep link to full PubMed search so it can be edited and re-run

	<br>Pro: Allows consulting other systems

	<br>Con: Makes it look like that consulting other systems is usual and necessary, rather than for rare needs.

<tr><td>Missing because terms appear in body, not title and abstract<td>Provide link to PubMed with full-text search

	<br>Pro: Allows expanding to full-text search without needing a reverse index.

	<br>Con: Harder to use than clicking a checkbox and re-running

<tr><td>2. Too many results<td>Imposed ordering of paginated results<td>Present unranked, unpaginated results under categories. See Note 2

	<br>Pro: Makes it easy to use the full result set

	<br>Con: Requires using the full, unranked result set.



<tr><td rowspan=2>3. Unknown confidence in results<td>Accepting narrow search results that look good<td>Human-in-the-loop to explore "what could I be missing" using designs to avoid missed sources in 1.3

	<br>Pro: Human in the loop is good control	

	<br>Con: Training is necessary to get consistent results across people.

<tr><td>Lack of consensus on source quality<td>Show all results that match, without imposing filtering

	<br>Pro: This is not a "solvable" problem. Humans have individual opinions. There should be deep links from results to pages with mechanisms to encourage easy customization and re-resulting with different source filters.

	<br>Con: Non-experts will get different results and take extra work to compare.

<tr><td colspan=3>Note 1: The importance of considering date in evaluating results cannot be over-emphasized given two facts:<ol><li>In the abstract of the Review article <A href="https://pubmed.ncbi.nlm.nih.gov/32166607/">A Review of Coronavirus Disease-2019 (COVID-19)</A>", (2020/03) the authors report "[SARS-CoV-2] spreads faster than its two ancestors the SARS-CoV and Middle East respiratory syndrome coronavirus (MERS-CoV), but has lower fatality." <li>The CORD-19 dataset includes search terms for MERS and SARS and since SARS-CoV-2 is so new, there are likely to be a disproportionate majority of articles which could answer CORD-19 questions unreliably.

</ol>NLP and automated text processing that estimates consensus will be working with a very large data set of possibly inappropriate sources. SARS-CoV-2 must is distinct from other coronaviruses at least in several important aspects. General conclusions made in 2019 about "coronavirus" will need to be re-evaluated and not used indiscriminately. In a research field that is so new, a human would deem the consideration of date of publication as an essential part of identifying trustworthy sources. Using date-range descending order of presenting a list of results is a simple way to help with this task.

<tr><td colspan=3>Note 2: The problem of too many search results is so common that we have become comfortable with partial and risky coping strategies. The most common is that some form of automatic ranking is used to show results in a sorted list with pagination. This is bad because even if all users agreed on the ranking criteria, it is not within the current capabilities of automated text processing systems to reliably evaluate sources for useful criteria, such as "trustworthiness." There are many criteria that might seem to be proxies for trustworthiness (such as type of publication systematic review vs. study, number of citations) they are not reliable for many reasons, including: 

<ul><li>dependence on reliable metadata creation. (There are gaps and limitations in the MEDLINE metadata.)

<li>different users will have different ranking criteria they would prefer. What most people would expect would be a composite multi-faceted analysis of several base criteria that are not consistent across all the articles in a large data set.

<li>in "cutting edge" research, there has not been enough time to establish a body of research with scientific consensus so new and old articles need different criteria applied.

<li>it is not simple to indicate the valuation of trustworthiness or other ranking criteria to a user, even if it were reliable. 

</ul>

 In most systems, the ranking criteria is imposed, not selected by the individual. This is risky because sorted results and pagination train and encourage people to be satisfied with looking through some of the results and stopping when they find something "sufficient" rather than "better" and "optimal." When the answer is critical, all the relevant results should be reviewed, and if filtering and ranking is to be employed, it should be directed by the user.

<p>The CORD-19 dataset has about 50,000 items, which is certainly too many items to display if a stop-word (or word from the orginal search criteria) is used for filtering.  But if articles are segregated by MeSH subject heading, the MeSH heading with the largest number of articles in the CORD-19 data set is Coronavirus Infections, with 2,485 items. Most users will want to apply criteria or at least one additional text search term to limit the set. But even if they do not, the 9.6MB metadata file and use of modern desktop web browsers easily show 2,485 in one list.

</table>



"""



h = display(HTML(htmlpros))
htmltask1a="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

Searching for (incubation period) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</p><p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

</p><p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

</p></blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tbody><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

</tr><tr valign="top"><td rowspan="4"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

</td></tr><tr valign="top"><td>

 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...The median <b>incubation</b> <b>period</b> was 6.7 days, the interval time from between illness onset and seeing a doctor was 4.5 days.

</td><td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign="top"><td>

 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/31995857">

Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia.

</a>

<small>(PMID31995857</small>)

<br>...The mean <b>incubation</b> <b>period</b> was 5.2 days (95% confidence interval [CI], 4.1 to 7.0), with the 95th percentile of the distribution at 12.5 days.

</td><td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign="top"><td>

 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32046816">

Effectiveness of airport screening at detecting travellers infected with novel coronavirus (2019-nCoV).

</a>

<small>(PMID32046816</small>)

<br>...In our baseline scenario, we estimated that 46% (95% confidence interval: 36 to 58) of infected travellers would not be detected, depending on <b>incubation</b> <b>period</b>, sensitivity of exit and entry screening, and proportion of asymptomatic cases.

</td><td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign="top"><td rowspan="2"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

</td></tr><tr valign="top"><td>

 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32046819">

Incubation period of 2019 novel coronavirus (2019-nCoV) infections among travellers from Wuhan, China, 20-28 January 2020.

</a>

<small>(PMID32046819</small>)

<br>...Using the travel history and symptom onset of 88 confirmed cases that were detected outside Wuhan in the early outbreak phase, we  estimate the mean <b>incubation</b> <b>period</b> to be 6.4 days (95% credible interval: 5.6-7.7), ranging from 2.1 to 11.1 days (2.5th to 97.5th percentile).

</td><td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign="top"><td rowspan="2"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

</td></tr><tr valign="top"><td>

 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...The disease is transmitted by inhalation or contact with infected droplets and the <b>incubation</b> <b>period</b> ranges from 2 to 14 d.

</td><td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

</tbody></table>

</p><hr><p>There are also 131 matches before 2019/12

</p>

"""



h = display(HTML(htmltask1a)) 
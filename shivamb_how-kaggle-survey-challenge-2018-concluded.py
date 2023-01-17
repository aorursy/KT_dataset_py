import pandas as pd 
import os

from IPython.display import display, HTML
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
from collections import Counter 

init_notebook_mode(connected=True)

kernel_slugs = """/headsortails/what-we-do-in-the-kernels-a-kaggle-survey-story
/sudalairajkumar/where-do-people-learn-ml-ds
/robikscube/a-tale-of-4-kaggler-types-by-ide-use-2018-survey
/andresionek/what-makes-a-kaggler-valuable
/harriken/storytelling-the-2018-kaggle-survey
/artgor/russia-usa-india-and-other-countries
/farazrahman/woggler-the-women-kaggler
/erikbruin/r-vs-python-and-kmodes-clustering-2018-survey
/ogakulov/the-mooc-wars-kaggle-s-perspective
/paultimothymooney/2018-kaggle-machine-learning-data-science-survey
/gaborfodor/are-you-a-data-scientist-no-way-hell-yeah
/ambarish/ml-kaggler-types-using-kmeans-and-pca
/arjundas/yet-another-data-science-story
/ash316/kaggle-journey-2017-2018
/mullervilmos/essential-data-skills-supply-and-demand
/pleonova/how-many-ml-frameworks-do-data-scientists-know
/umeshnarayanappa/the-world-needs-kaggle
/parvathykk/definitelydatascientists
/ambarish/a-forty-kaggler
/carlossouza/comparing-kaggle-and-stackoverflow-communities
/carlolepelaars/a-lighthearted-kaggle-2018-survey-eda
/shelars1985/analyzing-the-future-workforce
/nulldata/ml-bias-iml-perspective-recommendation
/amelthiarahel/is-data-science-for-me
/shubhammank/what-do-data-scientists-earn-world-wide
/vfdev5/who-are-they-data-scientists
/hulan1991/survey-analysis-choose-the-right-job
/aayush9876/you-india-you-lose
/gsdeepakkumar/who-is-a-data-scientist-a-statistical-approach
/alexdh359/top-earning-self-described-data-scientists
/kabure/understanding-kaggle-users-features
/aakashnain/is-it-better-than-2017
/danilodiogo/top-5-countries-us-india-china-russia-brazil
/ambarish/different-types-of-doctorate-kagglers
/madagasygirl/afrikagglers-the-african-kagglers
/gracefulibk/a-survey-story-of-data-chicks-on-kaggle
/ashishpatel26/sexiest-job-of-21st-century-data-scientist
/abhishekmamidi/insights-of-kaggle-ml-and-ds-survey
/niyamatalmass/how-to-become-a-data-scientist-in-2018
/youhanlee/find-recent-trends-of-kagglers-under-30-years-old
/seshadrikolluri/kaggle-survey-all-histograms-in-10-lines-of-code
/blazethrower/dkaggle-dt-the-change-of-kaggle-over-time
/garlsham/data-scientists-vs-data-analysts
/farazrahman/analysis-of-kaggle-products
/sikolia/kagglers-the-gender-story-a-2018-survey
/arunkumarramanan/deep-learning-awesome-resources
/dianakolusheva/machine-learning-self-starters
/algorrt/extensive-eda-sexiest-job-of-21st-century
/janani90/wrangling-kaggle-ml-and-ds-survey-data-using-r
/dhimananubhav/data-scientists-in-germany
/francescorivano/bandwagon-simulator
/nulldata/kaggle-survey-2018-the-girl-power
/christinampoid/which-data-career-should-you-follow
/kaiqidong/insights-in-different-career-paths-from-kaggle
/mhajabri/africai
/specbug/what-s-in-a-data-scientist-s-backpack
/pedroschoen/what-skills-move-up-our-compensation
/deepanshkhurana/kaggle-survey-is-there-someone-else-like-me
/ash316/ama-with-a-data-scientist
/bloodrabz/rich-kaggler-poor-kaggler
/lytyakov/kaggler-from-india-who-are-you
/eavdeeva/country-pay-gap
/vbrodrigues/what-tools-to-learn-as-a-ml-beginner
/anuraglahon/investigating-about-the-country-india
/nksingh673/peeping-into-organisations
/anammocanu/master-s-or-phd-in-data-science
/antgoldbloom/2018-kaggle-survey-eda-platforms-tools-and-more
/robikscube/2018-kaggle-survey-starter-kit
/arunkumarramanan/machine-learning-ml-awesome-frameworks
/felsal/kaggle-around-the-world
/debdutta/the-relatively-rich-data-scientists
/vpatricio/kagglers-in-context-country-income-group-analysis
/subversive/philippine-kagglers-2018
/rblcoder/2018-kaggle-survey-eda-platforms-tools-and-more
/graeme16161/who-is-a-data-scientist
/gpreda/data-scientists-in-2018-kaggle-survey
/slmf1995/kaggle-survey-2018-life-as-a-tech-recruiter
/sudhirnl7/data-science-survey-2018
/kerneler/starter-2018-kaggle-ml-ds-survey-7f4b06c8-4
/rangmar/simple-analysis-with-the-world-bank-data
/cegallo2/professional-development-in-2018-kaggle-survey
/lucifer19/programming-languages-vs-responders
/justjun0321/kickoff-with-basic-graphs
/seshadrikolluri/non-cs-phds-in-data-science-a-deep-dive
/yarnedia/clearing-a-path-to-a-six-figure-salary-in-the-us
/graeme16161/fairness-and-bias-in-machine-learning
/nizasiwale/kagglers-in-the-developing-and-developed-world
/arunkumarramanan/what-we-do-in-the-kernels-a-kaggle-survey-story
/hakkisimsek/plotly-tutorial-5
/arunkumarramanan/the-mooc-wars-kaggle-s-perspective
/datapsycho/the-skill-we-need-for-a-job
/mikewm24/crossroads-of-generation-x-and-millenial-kagglers
/chrispr/interview-with-a-data-scientist-or-not
/arunkumarramanan/ml-kaggler-types-found-using-unsupervised-learning
/paultimothymooney/2018-survey-data-specialists-from-the-usa
/ananthu017/kaggle-survey-for-people-new-to-data-science
/statmaster/exploring-kaggle-survey-2018-using-pandas
/anu2analytics/datascience-users-india-vs-us-men-vs-women
/jayesh4520/kaggle-survey-2018-simple-exploration
/vanshjatana/kagglers-across-the-world
/mjamilmoughal/valuable-insights-of-kagglers
/madagasygirl/industries-and-machine-learning
/hidelloon/building-data-analytics-center-to-where
/strangemane/measuring-accountability-in-ds-and-ml-with-waffles
/prashantgupta/once-upon-a-survey
/graeme16161/importance-of-interpretability
/shadabhussain/kaggle-storytelling
/arunkumarramanan/where-do-people-learn-ml-ds
/ambarish/a-high-paid-kaggler
/madagasygirl/tempus-fugit-from-2017-to-2018
/ihordurnopianov/what-we-do-is-who-we-are
/shamalip/kaggle-survey-for-the-aspirants
/deepak525/dss18
/joehabel/kagglers-without-a-bachelor-s
/docxian/kaggle-ml-ds-survey-multiple-choice-questions
/divyeshardeshana/data-scientist-role-analysis-with-34-graphs-more
/aleamva/the-kingdom-of-kaggle
/vanshjatana/kaggle-survey-visualisation
/arunsankar/key-insights-from-2018-kaggle-survey
/datark1/data-science-in-the-european-union-survey
/sank3t/kaggle-survey-usa-and-india
/arunkumarramanan/a-tale-of-4-kaggler-types-by-ide-use-2018-survey
/anupritad/ml-attracting-all-income-groups
/statnmap/maps-of-languages-and-analyses-reproducibility
/hiralmshah/ds-ml-survey-visualisation
/josephgpinto/data-beats-emotions
/hungfei/job-hunting-data-science-related
/rubmanoid/programming-language-and-coding-time
/mattinjersey/study-of-correlations
/kaiqidong/all-you-need-to-know-about-hottest-jobs
/martinlbarron/the-gender-divide-in-data-science
/shubhamlekhwar/data-science-a-hot-cake-in-industries
/victorlopez/gender-pay-gap-a-sad-and-true-story
/karthikdutt/13-questions-and-inferences-from-kaggle-survey
/harriken/residuals-fig8b-test
/ashkat/through-the-lens-of-gender-age-and-industry
/anupritad/ml-kaggle-attracting-all-socio-economic-groups
/hamelg/learning-data-science-the-role-of-online-courses
/kkurek012514/who-s-making-the-most-money
/gabrielmsilva/where-is-the-data-scientist
/ibtesama/india-usa-difference-in-data-science-scenario
/aamster/overview-of-different-types-of-data-scientists
/mamczurmiroslaw/quo-vadis-a-story-of-young-data-scientist
/duttadebadri/is-degree-mandatory-to-become-a-data-scientist
/ergitikajain/kagglers-then-now
/doha2012/how-much-can-i-earn
/marlukyanova/students-and-nonstudents-preferences-in-learning
/chrispr/gender-age-and-title-d3tree
/antoninaarefina/prefer-not-to-say-answers
/sangarshanan/answering-questions-with-the-survey-data
/mmfb65/2018-kaggle-survey
/ashwinids/toolset-of-a-data-scientist
/alijs1/how-to-earn-more-as-a-data-scientist
/ghost185/plot-for-basically-every-question-probably
/zgeek3/united-states-data-science-salaries
/ol0fmeister/the-student-s-hub
/marchman/2018-kaggle-ds-survey-clustering
/majickdave/ds-survey
/wladeczek44/90-of-free-form-questions-are-unanswered
/harriken/kaggle-journey-2017-2018
/jasonduncanwilson/kaggle-ml-survey-exploration
/piterfm/kaggle-ml-ds-survey-2018-who-are-kagglers
/amnfirst/the-data-science-line-how-to-do-it-well
/elilomeli/kagglers-in-the-developing-countries
/jeffysonar/diversity-around-world-age-and-free-responses
/yvancho/the-story-of-a-frustrating-realization
/hasanlianar/switchers-from-business-discipline-vs-others
/muhsina/road-to-data-scientist-without-cs-major
/kobespam/data-scientist-in-government-and-ngos
/toshinoue/kaggler-wage-index-2018
/nikitsoftweb/five-ways-to-quantify-yearly-compensation
/danialnam/library-to-help-your-kaggle-survey-challenge
/preductor/prospects-for-novices-in-data-science
/tombresee/2018-odst
/laloromero/a-brief-story-about-a-survey
/sanikamal/investing-the-country-indonesia
/harupy/favorite-media-sources-on-data-science
/lkuen89/most-valued-skill-stakeholder-communication
/geofizx/2018-kaggle-survey-inferring-respondent-personas
/mehmetcekic/ml-ds-getting-more-popular
/bishnuch/data-viz-on-kaggle-2018-survey-challenge
/supchanda08kol50/data-analysis-for-annual-data-science
/tannistha/what-are-the-top-software-used-by-data-scientist
/mgiraygokirmak/ey-ay
/holy185521/the-kagglers-by-region
/sherryxue/kaggle-ml-ds-survey-eda-on-who-responded
/mtmeanmachine/who-and-what-makes-data
/marchman/2018-ds-survey-q5
/tandonarpit6/non-data-scientists-leveraging-ml-ds
/allanray21/becoming-a-machine-learning-engineer
/sebastianpb10/can-your-yearly-compensation-be-predicted
/fthissite/studying-gender-disparities-among-kagglers
/amitprasad/r-and-python-who-s-the-difference
/theoviel/kagglers-gender-pay-gap-salary-prediction
/tinkudas/more-machine-learning-jobs-are-coming
/dave1216/data-wars
/heena34/how-to-grow-data-science-career
/gonnel/which-notebook-should-you-use-jupyter-or-rstudio
/rdrsquared/the-lens-through-which-we-see-the-world
/ustyk5/ukrainian-data-scientist-answers
/s3rg388/data-science-tutorial-income-gender-gap
/riteshsinha/the-job-of-a-data-scientist
/filthyilliterate/transitioning-from-academia-to-industry
/statmaster/the-data-science-gap-a-modeling-approach
/rajeshcv/overall-results-comparison-with-chosen-country
/sanwal092/gender-education-and-salaries
/slickwilly/meta-kernel-superba
/mkariithi/a-deep-dive-into-the-usage-of-r-python-or-both
/mikedev/kaggle-from-a-cs-master-degree-student-perspective
/cnavarrete72/fork-of-should-i-become-a-data-scientist
/cristianmb98/cristianborja-final
/salazar12/unsupervised-analytics-of-kagglers-in-2018
/nataliacp/ks-ncp-fi
/cferner/python-coders
/jinnies/what-makes-a-confident-data-scientist-in-the-u-s
/pchlq82/survey-analysis
/juan1996/la-comunidad-de-kaggle
/beatrizo96/kaggle-meta-challenge
/rnehas/descriptive-analysis-kaggle-survey-2018
/dcanulr/code-at-school
/yypark/established-vs-starting-data-scientists
/rubylime/kernel40f6ec4c03
/georgipetkov/beware-the-self-selection-bias
/sreelathar/story-told-by-graphs
/adityamanna/2018-kaggle-ml-ds-survey-analysis
/mkariithi/what-makes-a-kaggler-valuable
/mkariithi/where-do-people-learn-ml-ds
/pallavikhedekar/what-kaggle-wants-to-know
/shankhadeepghosal/2018-kaggle-ml-ds-survey-challenge
/jav1d98/kernel5cb14bd4cc
/shravankoninti/eda-and-insights-from-2018-kaggle-survey
/rajeshsupreet/2018-data-science-survey-story-telling
/maqbool/the-mooc-wars-kaggle-s-perspective
/slickwilly/2018-survey-kernel-path-generator
/rajthavti/who-is-a-data-scientist-a-statistical-approach
/wlkoch/2018-kaggle-ds-and-ml-survey-competition
/twalen/kaggle-data-science-survey-questions-map
/edimaudo/the-ds-ml-kaggle-story
/spscientist/2018-kaggle-machine-learning-data-science-survey
/milak303842/professional-online-learners-on-kaggle"""
kernel_slugs = kernel_slugs.split("\n")




user_images = """https://storage.googleapis.com/kaggle-avatars/thumbnails/1014468-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/71388-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/644036-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1549225-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/29522-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/727004-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/557776-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1443335-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/216754-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1314380-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/18102-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/103225-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/910033-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/740429-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1370076-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/139010-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/256944-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1552647-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/103225-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/915913-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1956324-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1211847-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/372183-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2396428-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/365460-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/425159-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2361804-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1106296-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1838666-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1097838-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/536977-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/435023-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/103225-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1564291-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2065763-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/859104-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1331984-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1115104-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1155353-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1061767-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1490279-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1365869-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/557776-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1772635-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1249072-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1045554-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/685355-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/372183-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/260853-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1987075-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1104210-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1470195-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2078060-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/740429-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1579096-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1245160-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1198164-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1858018-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1008767-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1033158-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/497585-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/368-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/644036-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/774828-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/262604-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1585022-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/185772-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1873860-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1331995-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/769452-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/807222-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/953273-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2080166-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1538816-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/2457001-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1142840-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1055009-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1061767-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2196481-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1331995-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1097890-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1544989-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1422342-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1078678-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1314380-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1384515-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2225966-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/494922-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2107080-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1670631-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1547362-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1564291-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1795159-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1741414-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1331995-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/681869-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/103225-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1564291-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1464916-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2242065-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2050543-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2031500-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1330628-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1968774-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2524872-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1670631-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/153258-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1144502-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2243081-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/949803-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2202938-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2471419-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1584393-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/511687-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/828098-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2283470-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1987075-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/40730-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2401028-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/226229-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/716649-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/29522-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2202938-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/183041-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1339681-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2167966-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1611087-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1365835-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1341160-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/859381-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/615583-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1078678-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1288337-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1512902-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1408504-kg.PNG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1844623-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1311374-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2095225-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/592643-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1336564-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/29522-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1376207-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/861427-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/539747-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1687011-kg.jpeg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/772965-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2443608-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/530297-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1327481-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1938841-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2130345-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1663348-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1272482-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1913939-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2350786-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/885509-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1460211-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/185521-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2195635-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2095225-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1976308-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/506442-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2438555-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/644147-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2062758-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2390426-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/209391-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/647743-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1352412-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2343674-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/104367-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2225966-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1187996-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/1734142-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/826299-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/697622-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2437106-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1252796-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/2461421-kg.PNG
https://storage.googleapis.com/kaggle-avatars/thumbnails/2520699-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/447337-kg.JPG
https://storage.googleapis.com/kaggle-avatars/thumbnails/2043577-gr.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/1997757-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2417770-fb.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/46947-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/585916-kg.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/826299-kg.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/2508495-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/1856194-gp.jpg
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png
https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png"""
user_images = user_images.split("\n")

def _load_dataset(filename):
    path = "../input/"
    filepath = path + filename
    dataframe = pd.read_cv(filepath)
    return dataframe

def _pre_process(dataframe):
    df = dataframe
    columns = df.columns
    print (columns)
    print (df.info())
    print (df.head())
    
def value_counts(df, col):
    vc = df[col].value_counts()
    return vc

def aggregate(df, col, agg_col):
    agg = df.groupby(col).agg({agg_col : "mean"})
    agg = agg.reset_index()
    agg = agg.sort_values(agg_col, ascending = False)
    return agg


# load data
kernels = pd.read_csv("../input/meta-kaggle/Kernels.csv")
Users = pd.read_csv("../input/meta-kaggle/Users.csv")
KernelVersions = pd.read_csv("../input/meta-kaggle/KernelVersions.csv")
KernelVotes = pd.read_csv("../input/meta-kaggle/KernelVotes.csv")
KernelTags = pd.read_csv("../input/meta-kaggle/KernelTags.csv")

## survey data 
os.listdir("../input/kaggle-survey-2018/")
survey = pd.read_csv("../input/kaggle-survey-2018/SurveySchema.csv")


# preprocessing
survey_kernels = []
for kernel in kernel_slugs:
    slug = kernel.split("/")[2]
    username = kernel.split("/")[1]

    slugged = kernels[kernels["CurrentUrlSlug"] == slug]
    slugged = slugged[slugged["ForkParentKernelVersionId"].isna()]
    if (len(slugged) != 0):
        doc = dict(slugged.head().iloc(0)[0])
        survey_kernels.append(doc)

kernels_df = pd.DataFrame(survey_kernels)
kernels_df["CreationDate"] = pd.to_datetime(kernels_df["CreationDate"])
kernels_df["date"] = "2018-" + kernels_df["CreationDate"].dt.month.astype(str) + "-" + kernels_df["CreationDate"].dt.day.astype(str)
kernels_df["day"] = kernels_df["CreationDate"].dt.day
Users = Users.rename(columns={"Id" : "AuthorUserId"})
kernels_df = kernels_df.merge(Users, on = "AuthorUserId")
kernels_df = kernels_df.drop_duplicates()
visited = {}
html = "<div>"
for img in user_images:
    if img not in visited:
        visited[img] = 1
        
        if "default-thumb" in img:
            continue
        html += "<img src='" +img+ "' width='60px' style='float:left; margin-left:5px; border:1px solid black'>"
html += "</div>"
display (HTML( html))
teir_colors ={ 'Novice' : '#5ac995', 'Contributor' : '#00BBFF', 'Expert' : '#95628f', 'Master' : '#f96517', 'GrandMaster' : '#dca917', 'KaggleTeam' : '#008abb'}

vc1 = kernels_df.groupby("PerformanceTier").agg({"CurrentUrlSlug" : "count", "TotalVotes" : "sum", "TotalViews" : "sum"}).reset_index().rename(columns={"CurrentUrlSlug" : "count"})
mapp = {0:"Novice", 1 : "Contributor", 2 : "Expert", 3 : "Master", 4 : "GrandMaster", 5 : "KaggleTeam"}
vc1['teir'] = vc1["PerformanceTier"].apply(lambda x : mapp[x])
vc1['teir_col'] = vc1["teir"].apply(lambda x : teir_colors[x])

trace = go.Bar(x = vc1.teir, y = vc1["count"], marker = dict(color=vc1["teir_col"]))
layout = go.Layout(title="Kernels Published by Kaggler Tiers", height=400, yaxis=dict(title="Total Kernels", range=(0,100)))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
trace0 = go.Scatter(x = vc1["count"], y = vc1.TotalVotes, mode='markers+text',
    text = vc1.teir, marker=dict(color=vc1.teir_col, size=vc1.TotalViews*0.005, opacity=0.8))
layout = go.Layout(title="Cumulative Votes and Views (Size) Garnered by Kernels Tiers", 
                   xaxis = dict(title="Total Kernels"), yaxis = dict(title="Total Cumulative Votes"))
data = [trace0]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename='bubblechart-color')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
import base64

title = 'Number of Kernels by Date'
overdose = kernels_df["day"].value_counts().to_frame()
overdose = overdose.rename(columns = {"day" : title})
kernels_df['ndate'] = pd.to_datetime(kernels_df['date']).dt.normalize()
vc = kernels_df['ndate'].value_counts().to_frame().reset_index().rename(columns = {"ndate" : "counts", "index" : "date"})
vc = vc.sort_values("date")
vc['cum_counts'] = vc["counts"].cumsum()

fig = plt.figure(figsize=(10,6))
plt.xlim(min(vc.date), max(vc.date))
plt.ylim(min(vc.cum_counts), 210)
plt.xlabel('Date',fontsize=20)
plt.ylabel(title, fontsize=20)
plt.title('Total Kernels by Date', fontsize=20)

def animate(i):
    data = vc.iloc[:int(i+1)] 
    p = sns.lineplot(x=data.date, y=data.cum_counts, data=data, color="#f4aa42")
    p.tick_params(labelsize=15)
    plt.setp(p.lines, linewidth=5)

ani = animation.FuncAnimation(fig, animate, frames=len(vc), repeat=True)

ani.save('test.gif', writer='imagemagick', fps=10)
plt.close(1)
filename = 'test.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
trace1 = go.Bar(x = vc.date, y = vc.counts, name = "New Kernels on a particular day", marker = dict(color="red", opacity=0.8))
trace2 = go.Scatter(x = vc.date, y = vc.cum_counts, name="Total Kernels on a particular day", marker = dict(color="orange", opacity=0.8), mode = "lines+markers")
layout = go.Layout(title="Kernels Published by Date", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data = [trace1, trace2], layout = layout)
iplot(fig)
py_r = ['Rmd', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'R', 'Rmd', 'Py', 'Py', 'Py',\
 'Rmd', 'R', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'Rmd', 'Py', 'Py', 'Py', \
 'Rmd', 'Py', 'Py', 'Py', 'R', 'Py', 'Rmd', 'Py', 'Py', 'Py', 'R', 'Rmd',\
  'Rmd', 'Py', 'Py', 'Py', 'Py', 'Py', 'R', 'Py', 'Py', 'R', 'Rmd', 'Py', \
  'Py', 'Py', 'R', 'Py', 'Py', 'Rmd', 'Py', 'Py', 'Py', 'Py', 'Py', 'R', \
  'Py', 'Py', 'Py', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'Py', 'Py', \
  'Py', 'Py', 'Rmd', 'Py', 'R', 'Py', 'Rmd', 'Py', 'Rmd', 'Py', 'Py', 'Py',\
   'Py', 'Py', 'Py', 'R', 'R', 'Rmd', 'Py', 'Rmd', 'Py', 'Py', \
  'Py', 'Py', 'Rmd', 'Rmd', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'Py', 'Py', 'Rmd',\
   'Py', 'Rmd', 'Py', 'Rmd', 'Py', 'Py', 'Rmd', 'Rmd', 'Py', \
  'Py', 'Py', 'Py', 'R', 'Py', 'Py', 'Rmd', 'Py', 'Py', 'Py', 'Py', 'Py',\
   'Rmd', 'Py', 'Py', 'Py', 'Py', 'R', 'Py', 'Rmd', 'Py', 'Py',\
   'Py', 'Rmd', 'Py', 'Py', 'R', 'Py', 'Rmd', 'Py', 'Rmd', 'Py', 'Py', 'Py',\
    'Py', 'Py', 'R', 'Py', 'Py', 'R', 'Py', 'Py', 'Py', 'Py', 'Py', 'Py', \
    'Py', 'Py', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'Py', 'Py', 'Py', 'Rmd', \
    'Py', 'Rmd', 'Py', 'Py', 'Py', 'Py', 'R', 'Py', 'Py', 'Py', 'Py',\
     'Py', 'Py', 'Py', 'Py', 'Py', 'Py', 'R', 'Py', 'Py', 'Py', 'Py', 'Py',\
      'Py', 'R', 'Py', 'Py', 'Rmd', 'Py', 'R', 'Rmd', 'Rmd', 'Py', 'R', 'Py', \
      'Py', 'Py', 'Py', 'R', 'Rmd', 'Py', 'Py', 'Py', 'Py', 'Py', 'Py', 'Rmd', \
      'Rmd', 'Py', 'Py', 'Py', 'Py', 'Rmd', 'Py', 'R', 'Py', 'Py', 'Py', 'Py',\
       'Py', 'Py', 'Py', 'Py', 'Py', 'Py', 'R', 'Rmd', 'R', 'Py', 'R', 'Py', 'Py']

    

from collections import Counter 
py_r = ["R" if _ == "Rmd" else _ for _ in py_r]

py_r_dist = Counter(py_r).most_common(5)
# print (py_r_dist)
labels = [x[0] for x in py_r_dist]
values = [x[1] for x in py_r_dist]
pitrace = go.Pie(labels = labels, values = values, marker = dict(colors = ["#cfefac", "#ed50aa"]))

layout = go.Layout(title = "Python or R Kernels")
fig = go.Figure(data =[pitrace], layout = layout)
iplot(fig)
kernels_df["Medal"] = kernels_df["Medal"].fillna(4)
medal_col = { 1 : "#FFCE3F", 2 : "#E6E6E6", 3 : "#EEB171" , 4 : "#a2c3f9"}
medal_name = { 1 : "Gold", 2 : "Silver", 3 : "Bronze" , 4 : "NoMedal"}

vc = kernels_df.groupby("Medal").agg({"TotalViews" : "sum", "CurrentUrlSlug" : "count"}).reset_index()
vc["medal_col"] = vc['Medal'].apply(lambda x : medal_col[x] if x in medal_col else "#222")
vc["medal_name"] = vc['Medal'].apply(lambda x : medal_name[x])

trace1 = go.Bar(x = vc.medal_name, y=vc.CurrentUrlSlug, name="Total Kernels", orientation = "v" ,
                   marker=dict(color=vc.medal_col, opacity=1.0))
trace2 = go.Bar(x = vc.medal_name, y=vc.TotalViews, name="Total Views", orientation = "v" ,
                   marker=dict(color=vc.medal_col, opacity=1.0))

fig = tools.make_subplots(rows=1, cols=2, print_grid = False, subplot_titles=["Total Medals Count", "Total Views by Medals"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(title='', height=400, showlegend = False)
iplot(fig)
kernels_df['teir'] = kernels_df["PerformanceTier"].apply(lambda x : mapp[x])
kernels_df['teir_col'] = kernels_df["teir"].apply(lambda x : teir_colors[x])

kernels_df["medal_col"] = kernels_df['Medal'].apply(lambda x : medal_col[x] if x in medal_col else "#222")


trace0 = go.Scatter(
    x=kernels_df.TotalComments,
    y=kernels_df.TotalVotes,
    mode='markers',
    text = kernels_df.CurrentUrlSlug,
    marker=dict(
            color= kernels_df.teir_col,
        opacity=0.8,
        size=kernels_df.TotalViews*0.01,
    )
)

data = [trace0]
layout = go.Layout(xaxis=dict(title="Total Comments"), yaxis=dict(title="Total Votes"), title="Popular Kernels by Votes, Comments, Views")
fig = go.Figure(layout = layout, data = data)
iplot(fig, filename='bubblechart-color')
trace0 = go.Scatter(
    x=kernels_df.TotalComments,
    y=kernels_df.TotalVotes,
    mode='markers',
    text = kernels_df.CurrentUrlSlug,
    marker=dict(
            color= kernels_df.medal_col,
        opacity=1,
        size=kernels_df.TotalViews*0.01,
    )
)

data = [trace0]
layout = go.Layout(xaxis=dict(title="Total Comments"), yaxis=dict(title="Total Votes"), title="Popular Kernels by Votes, Comments, Views")
fig = go.Figure(layout = layout, data = data)
iplot(fig, filename='bubblechart-color')
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

title_words = "-".join(kernels_df.CurrentUrlSlug).split("-")
title_text = " ".join(title_words).lower()

ignorewords = ["kaggler", "kaggle", "data", "scientist", "survey", "science", "ml", "ds", "analysis", "story", " s ", "2018", "2017", "machine", "learning"]
for common in ignorewords:
    title_text = title_text.replace(common, " ")

# wc = WordCloud(stopwords=STOPWORDS, colormap='cool', background_color='#fff').generate(title_text)
# plt.figure(figsize=(12,12))
# plt.imshow(wc)
# plt.axis('off')
# plt.title('WordCloud : All Words Used');
def _get_version_values(x):
    versions = KernelVersions[KernelVersions['KernelId'] == x]
    versions = versions[~versions['VersionNumber'].isna()]
    total_runs = len(versions)
    version_lines = list(versions['TotalLines'])
    version_votes = list(versions['TotalVotes'])
    
    changes = [x - version_lines[i - 1] for i, x in enumerate(version_lines)][1:]
    
    total_lines = version_lines[-1]
    avg_votes_pv = sum(version_votes) / total_runs
    avg_lines_pv = sum(changes) / (total_runs)

    return total_runs, total_lines, avg_votes_pv, avg_lines_pv


kernels_df['total_version_runs'] = kernels_df['Id'].apply(lambda x : _get_version_values(x)[0])
kernels_df['total_lines'] = kernels_df['Id'].apply(lambda x : _get_version_values(x)[1])
kernels_df['avg_votes_pv'] = kernels_df['Id'].apply(lambda x : _get_version_values(x)[2])
kernels_df['avg_lines_pv'] = kernels_df['Id'].apply(lambda x : _get_version_values(x)[3])

t1 = kernels_df.groupby("DisplayName").agg({"total_lines" : "mean"}).reset_index()
t1 = t1.merge(kernels_df[["DisplayName", "teir_col"]], on="DisplayName").drop_duplicates()
t1 = t1.sort_values("total_lines", ascending = False).head(25)[::-1]

trace = go.Bar(y = t1.DisplayName, x = t1.total_lines, orientation="h", marker = dict(color=t1.teir_col, opacity=0.8))
layout = go.Layout(title="Kernels with Maximum Lines of Content", height=600, margin=dict(l=180))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
t1 = kernels_df.groupby("DisplayName").agg({"total_version_runs" : "sum"}).reset_index()
t1 = t1.merge(kernels_df[["DisplayName", "teir_col"]], on="DisplayName").drop_duplicates()
t1 = t1.sort_values("total_version_runs", ascending = False).head(25)[::-1]

trace = go.Bar(y = t1.DisplayName, x = t1.total_version_runs, orientation="h", marker = dict(color=t1.teir_col, opacity=0.8))
layout = go.Layout(title="Kernels with Maximum Versions", height=600, margin=dict(l=180))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
tags = ['storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'classification', 'advanced', 'eda', 'data cleaning', 'data visualization', 'linear regression', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'survey analysis', 'storytelling', 'data visualization', 'survey analysis', 'data visualization', 'starter code', 'data cleaning', 'data visualization', 'classification', 'feature engineering', 'tutorial', 'data visualization', 'clustering', 'k-means', 'pca', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'beginner', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'feature engineering', 'eda', 'data visualization', 'survey analysis', 'feature engineering', 'eda', 'data cleaning', 'data visualization', 'survey analysis', 'beginner', 'data visualization', 'eda', 'data cleaning', 'data visualization', 'survey analysis', 'tabular data', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'tutorial', 'beginner', 'data visualization', 'starter code', 'india', 'beginner', 'eda', 'data visualization', 'starter code', 'data visualization', 'classification', 'storytelling', 'beginner', 'data visualization', 'storytelling', 'beginner', 'eda', 'data visualization', 'starter code', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'categorical data', 'tutorial', 'eda', 'survey analysis', 'starter code', 'advanced', 'eda', 'data visualization', 'survey analysis', 'tabular data', 'data visualization', 'survey analysis', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'women', 'beginner', 'eda', 'data visualization', 'tutorial', 'beginner', 'eda', 'data visualization', 'dailychallenge', 'eda', 'data cleaning', 'data visualization', 'storytelling', 'tutorial', 'beginner', 'data visualization', 'survey analysis', 'tutorial', 'beginner', 'eda', 'data visualization', 'storytelling', 'beginner', 'data visualization', 'categorical data', 'data visualization', 'survey analysis', 'eda', 'survey analysis', 'research tools and topics', 'libraries', 'tutorial', 'beginner', 'deep learning', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'starter code', 'beginner', 'eda', 'data visualization', 'feature engineering', 'starter code', 'beginner', 'eda', 'data visualization', 'storytelling', 'beginner', 'data cleaning', 'data visualization', 'storytelling', 'eda', 'data cleaning', 'data visualization', 'survey analysis', 'storytelling', 'africa', 'beginner', 'data visualization', 'survey analysis', 'beginner', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'data visualization', 'survey analysis', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'eda', 'data visualization', 'classification', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'beginner', 'data visualization', 'survey analysis', 'data visualization', 'survey analysis', 'starter code', 'libraries', 'tutorial', 'beginner', 'feature engineering', 'deep learning', 'storytelling', 'eda', 'data cleaning', 'data visualization', 'data visualization', 'eda', 'survey analysis', 'asia', 'industry', 'eda', 'data visualization', 'beginner', 'eda', 'data visualization', 'data visualization', 'tutorial', 'beginner', 'eda', 'data visualization', 'starter code', 'eda', 'data visualization', 'classification', 'storytelling', 'tutorial', 'beginner', 'eda', 'data visualization', 'starter code', 'storytelling', 'beginner', 'beginner', 'eda', 'data visualization', 'survey analysis', 'data visualization', 'eda', 'data visualization', 'starter code', 'tutorial', 'beginner', 'survey analysis', 'classification', 'decision tree', 'data visualization', 'employment', 'economics', 'tutorial', 'eda', 'data visualization', 'storytelling', 'data visualization', 'survey analysis', 'storytelling', 'eda', 'data visualization', 'logistic regression', 'binary classification', 'storytelling', 'beginner', 'eda', 'data visualization', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'data journalism', 'tutorial', 'intermediate', 'data visualization', 'clustering', 'storytelling', 'beginner', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'feature engineering', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'data visualization', 'storytelling', 'beginner', 'data visualization', 'survey analysis', 'data visualization', 'dimensionality reduction', 'eda', 'data visualization', 'survey analysis', 'starter code', 'tutorial', 'beginner', 'data cleaning', 'data visualization', 'starter code', 'beginner', 'data visualization', 'beginner', 'data visualization', 'starter code', 'eda', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'starter code', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'classification', 'cartography', 'geospatial analysis', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'starter code', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'beginner', 'starter code', 'storytelling', 'eda', 'data visualization', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'data visualization', 'beginner', 'eda', 'data visualization', 'survey analysis', 'storytelling', 'data visualization', 'survey analysis', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'eda', 'data visualization', 'survey analysis', 'clustering', 'storytelling', 'data visualization', 'survey analysis', 'data visualization', 'survey analysis', 'beginner', 'eda', 'data visualization', 'beginner', 'data visualization', 'storytelling', 'tutorial', 'beginner', 'data visualization', 'survey analysis', 'beginner', 'eda', 'data visualization', 'beginner', 'survey analysis', 'clustering', 'data visualization', 'survey analysis', 'data visualization', 'eda', 'data visualization', 'storytelling', 'world', 'countries', 'eda', 'geospatial analysis', 'data visualization', 'data visualization', 'survey analysis', 'classification', 'data visualization', 'regression analysis', 'tutorial', 'beginner', 'data visualization', 'starter code', 'data visualization', 'gradient boosting', 'data cleaning', 'data visualization', 'survey analysis', 'clustering', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'data cleaning', 'data visualization', 'classification', 'starter code', 'eda', 'data visualization', 'starter code', 'eda', 'data visualization', 'storytelling', 'eda', 'data visualization', 'gradient boosting', 'storytelling', 'beginner', 'eda', 'data visualization', 'survey analysis', 'beginner', 'eda', 'data cleaning', 'data visualization', 'starter code', 'data visualization', 'statistical analysis', 'survey analysis', 'eda', 'data visualization', 'survey analysis', 'classification', 'storytelling', 'eda', 'data visualization', 'survey analysis', 'beginner', 'data cleaning', 'data visualization', 'beginner', 'data visualization', 'clustering', 'data cleaning', 'data visualization', 'survey analysis', 'eda', 'data cleaning', 'data visualization', 'eda', 'data visualization', 'eda', 'data cleaning', 'data visualization', 'data visualization', 'starter code']

vc = Counter(tags).most_common(25)
x = [_[0] for _ in vc]
y = [_[1] for _ in vc]

trace = go.Bar(x = x, y = y, marker = dict(color="darkslategray"))
layout = go.Layout(title="Popular Tags used by participants", height=400, yaxis=dict(title="Tag Count", range=(0,150)))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
t1 = kernels_df.groupby("UserName").agg({"TotalViews" : "sum", "TotalVotes" : "sum", "AuthorUserId" : "count"}).reset_index()
t1 = t1[t1["AuthorUserId"] > 1]

trace0 = go.Scatter(x = t1["TotalVotes"], y = t1["TotalViews"], mode='markers',
    text = t1["UserName"]+" (Kernels:"+t1['AuthorUserId'].astype(str)+" Votes: " +t1['TotalVotes'].astype(str)+ ")", marker=dict(color=t1.TotalVotes, colorscale="Jet", showscale = True, size=t1.TotalVotes*0.7, opacity=0.6))
layout = go.Layout(title="Kagglers who shared multiple kernels (Size : Total Votes)", 
                   xaxis = dict(title="Total Votes Got"), yaxis = dict(title="Total Views Got"))
data = [trace0]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename='bubblechart-color')
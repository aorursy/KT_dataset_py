# Questions and their queries

tasks = [

    {

        "text": "What is known about transmission, incubation, and environmental stability?",

        "questions": [

            {

                "text": "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

                "queries": [

                    {

                        "text": "Search: incubation, period, day, !cattle",

                        "comment": "We add ignore the snippets containing cattle as we care about human research",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '''incub'' & ''period'' & ''day'' & !''cattl''') &&

                                to_tsquery('coronavirus | COVID:* | SARS-CoV:*')

                                ,to_tsquery('english', 'incub | period | day')

                            )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

                "queries": [

                    {

                        "text": "Search: asymptomatic shedding, transmission",

                        "comment": "Ignore text, which mentions cows",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('''asymptomat'' & ''shed'' & ''transmiss'' & !''cattl'' & !''anim'' & !''calv''') &&

                                to_tsquery('coronavirus | COVID:* | SARS-CoV:*')

                                ,to_tsquery('english', 'asymptomat | shed | transmiss')

                            )

                            order by min nulls last

                            limit 5;

                        """

                    },

                    {

                        "text": "Search: children, transmission",

                        "comment": "Focus on the text mentioning children and transmissions",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('''children'' & ''transmiss'' & !''cattl'' & !''anim'' & !''calv''') &&

                                to_tsquery('coronavirus | COVID:* | SARS-CoV:*')

                                ,null

                            )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Seasonality of transmission",

                "queries": [

                    {

                        "text": "Search: seasonal, transmission",

                        "comment": "Ingnore transmission in animals",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('''season'' & ''transmiss'' & !''cattl'' & !''anim'' & !''calv''') &&

                                to_tsquery('coronavirus | COVID:* | SARS-CoV:*')

                                ,null

                            )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

                "queries": [

                    {

                        "text": "Search: charge distribution",

                        "comment": "Looking at charge distribution, while ignoring animal documents",

                        "search": """

                            select *

                            from snippets(

                                phraseto_tsquery('english', 'charge distribution') && to_tsquery('english', '! cattle & ! animal & ! calves') && to_tsquery('coronavirus | COVID:* | SARS-CoV:*')

                                ,plainto_tsquery('english', 'charge distribution')

                            )

                            order by min nulls last

                            limit 5; 

                        """

                    },

                    {

                        "text": "Search: viral shedding",

                        "comment": "Looking at viral shedding, with scoring based on min term distance",

                        "search": """

                            select *

                            from snippets(

                                phraseto_tsquery('english', 'viral shedding') && to_tsquery('english', '! cattle & ! animal & ! calves') && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , null

                                )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

                "queries": [

                    {

                        "text": "Search for nasal discharge, sputum, urine, fecal mater, blood",

                        "comment": "Scoring based on min distance between key terms",

                        "search": """

                            select *

                            from snippets(

                                (to_tsquery('english', '(nasal & discharge) | sputum | urine | (fecal & matter) | blood') &&

                                    to_tsquery('english', 'stability | persistence')) &&

                                to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*'),

                                to_tsquery('english', 'nasal | discharge | sputum | urine | fecal | matter | blood | stability | persistence')

                            )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

                "queries": [

                    {

                        "text": "Search for materials, copper, steel, plastic, virus persistance",

                        "comment": "Scoring based on min distance between key terms",

                        "search": """

                            select *

                            from snippets(

                                (to_tsquery('english', 'materials | copper | steel | plastic') &&

                                    to_tsquery('english', 'persistence & virus')) &&

                                to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*'),

                                to_tsquery('english', 'materials | copper | steel | plastic | virus | persistence')

                            )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Natural history of the virus and shedding of it from an infected person",

                "queries": [

                    {

                        "text": "Search for shedding, infected host",

                        "comment": "Scoring based on avg distance between key terms",

                        "search": """

                            select *

                            from snippets(

                                (to_tsquery('english', 'shedding & infected & host')) &&

                                to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*'),

                                to_tsquery('english', 'shedding | infected | host')

                            )

                            order by avg nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Implementation of diagnostics and products to improve clinical processes",

                "queries": [

                    {

                        "text": "Search for improve, clinical, process",

                        "comment": "Scoring based on avg distance between key terms",

                        "search": """

                            select *

                            from snippets(

                                (to_tsquery('english', ' improve & clinical & processes'))

                                ,null

                            )

                            order by avg nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Disease models, including animal models for infection, disease and transmission",

                "queries": [

                    {

                        "text": "Search for disease models",

                        "comment": "Looking disease models phrase, scoring based on average term distance",

                        "search": """

                            select *

                            from snippets(

                                phraseto_tsquery('english', 'disease models') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                ,to_tsquery('english', 'disease | models')

                                )

                            order by min nulls last

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Tools and studies to monitor phenotypic change and potential adaptation of the virus",

                "queries": [

                    {

                        "text": "Search for phenotypic change",

                        "comment": "Only 1 document found, no sorting needed",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '(phenotypic <-> change)') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , null

                                );

                        """

                    }

                ]

            },

            {

                "text": "Immune response and immunity",

                "queries": [

                    {

                        "text": "Immune response",

                        "comment": "Look for immune response associated with versions of coronavirus",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Immune & response') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                            --     , null

                                , to_tsquery('english', 'immune | response')

                                )

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

                "queries": [

                    {

                        "text": "Search for effectiveness of movement control strategies",

                        "comment": "Score by minimum distance between terms",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Effectiveness & movement & control') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                            , to_tsquery('english', 'Effectiveness | movement | control')

                                )

                            order by min

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for prevent secondary transmission",

                        "comment": "Score by minimum distance to find most relevant snippets",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'prevent & secondary & transmission') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                , to_tsquery('english', 'prevent | secondary | transmission')

                                )

                            order by min

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for transmission in health care",

                        "comment": "Scored by minimum distance and include coronavirus terms",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'transmission & health & care') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'transmission | health | care')

                                )

                            order by min

                            limit 10;

                        """

                    }

                ]

            },

            {

                "text": "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

                "queries": [

                    {

                        "text": "Search for effectiveness of (personal protective equipment) or acronym PPE",

                        "comment": "Scoring based on minimum distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Effectiveness & ((personal & protective & equipment) | PPE)') -- && to_tsquery('english', '! cattle & ! animal & ! calves')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'Effectiveness | personal | protective | equipment | PPE')

                                )

                            order by min

                            limit 10;

                        """

                    },

                    {

                        "text": "Search of effectiveness of PPE",

                        "comment": "Scored based on minimum distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '(reduce & risk & transmission) & ((personal & protective & equipment) | PPE)')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'reduce | risk | transmission | personal | protective | equipment | PPE')

                                )

                            order by min

                            limit 10;

                        """

                    }

                ]

            }

        ]

    },

    {

        "text": "What do we know about COVID-19 risk factors?",

        "questions": [

            {

                "text": "Data on potential risks factors",

                "queries": [

                    {

                        "text": "Search for risk factors, patients",

                        "comment": "Score by average term distance, include coronavirus terms",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '(risks & factors & patients)')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'risks | factors | patients')

                                )

                            order by avg

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for smoking or predisposing pulmonary disease",

                        "comment": "Sort by minimum distance, to make sure we find the phrase pulmonary disease",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Smoking | (predisposing & pulmonary & disease)')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'Smoking | pulmonary | disease | predisposing')

                                )

                            order by min

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for neonates, pregnant women",

                        "comment": "Order by min term distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Neonates & pregnant & women')

                            --          && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                                , to_tsquery('english', 'Neonates | pregnant| women')

                                )

                            order by min

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",

                "queries": [

                    {

                        "text": "Search for basic reproductive number",

                        "comment": "Return single result, as it seems to be the best answer",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'basic & reproductive & number')

                                     && to_tsquery('english', 'coronavirus | COVID:* | SARS-CoV:*')

                            --     , to_tsquery('english', 'Neonates | pregnant| women')

                                ,null

                                )

                            order by min

                            limit 1;

                        """

                    },

                    {

                        "text": "Search for serial interval days",

                        "comment": "Look for serial interval phrase followed by days. Sort by min term distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '(serial <-> interval) & days')

                                , to_tsquery('english', 'serial | interval | days')

                            --     ,null

                                )

                            order by min

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for incubation period days",

                        "comment": "Look for incubation period phrase followed by days, as well as including coronavirus terms. Sorted by min term distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', '(incubation <-> period) & days')

                                && to_tsquery('english', 'coronavirus | COVID:*')

                                , to_tsquery('english', 'incubation | period | days')

                            --     ,null

                                )

                            order by min

                            limit 5;

                        """

                    },

                    {

                        "text": "Search for modes of transmission",

                        "comment": "Scored by min term distance. Include coronavirus terms",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'modes & transmission')

                                && to_tsquery('english', 'coronavirus | COVID:*')

                                , to_tsquery('english', 'modes | transmission')

                            --     ,null

                                )

                            order by min

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",

                "queries": [

                    {

                        "text": "Search for severity, risk, fatality",

                        "comment": "Look for search + coronavirus terms and score by min distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'severity & risk & fatality')

                                && to_tsquery('english', 'coronavirus | COVID:*')

                                , to_tsquery('english', 'severity | risk | fatality')

                            --     ,null

                                )

                            order by min

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Susceptibility of populations",

                "queries": [

                    {

                        "text": "Search for susceptibility and populations",

                        "comment": "Look for search + coronavirus terms and score by min distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Susceptibility & populations')

                                && to_tsquery('english', 'coronavirus | COVID:*')

                                , to_tsquery('english', 'Susceptibility | populations')

                            --     ,null

                                )

                            order by min

                            limit 5;

                        """

                    }

                ]

            },

            {

                "text": "Public health mitigation measures that could be effective for control",

                "queries": [

                    {

                        "text": "Search for public health (measures | mitigation)",

                        "comment": "Look for search + coronavirus terms and score by min distance",

                        "search": """

                            select *

                            from snippets(

                                to_tsquery('english', 'Public & health & (measures | mitigation)')

                                && to_tsquery('english', 'coronavirus | COVID:*')

                                , to_tsquery('english', 'Public | health | measures | mitigation')

                            --     ,null

                                )

                            order by min

                            limit 5;



                        """

                    }

                ]

            }

        ]

    }

]
!pip install psycopg2-binary yattag
import psycopg2



connection = psycopg2.connect(user = 'data_access',

                              password = 'H5yfSApxS5bk6f',

                              host = 'spotlightdata-covid-do-user-1518235-0.a.db.ondigitalocean.com',

                              port = '25060',

                              database = 'covid')
# Set search path for the session

cur = connection.cursor()

cur.execute("set search_path to nanowire");

cur.close()

print("search_path set");
# Run all search queries and save the results

from psycopg2.extras import RealDictCursor

import time



def retrieve(query):

    cur = connection.cursor(cursor_factory=RealDictCursor);

    cur.execute(query);

    rows = cur.fetchall();

    cur.close();

    return rows



total = 0;

for ti, task in enumerate(tasks):

    for qi, question in enumerate(task["questions"]):

        total += len(question["queries"])



print("Calculated total queries: {}".format(total))

        

idx = 0;

for ti, task in enumerate(tasks):

    for qi, question in enumerate(task["questions"]):

        for qri, query in enumerate(question["queries"]):

            start = time.time()

            query["snippets"] = retrieve(query["search"]);

            end = time.time()

            idx += 1;

            print("[{}/{}] Task [{}], Question [{}], Query [{}] Done in: ({:.3f}s)".format(idx, total, ti + 1, qi + 1, qri + 1, end - start))



print("FINISHED");
# Create a html document containing the output

from yattag import Doc



def render_text(tag, text, raw_text):

    for i, snippet in enumerate(raw_text.split("#$#")):

        class_name = "marked" if i % 2 != 0 else ""

        with tag('span', klass=class_name):

            text(snippet)



def render_queries(tag, text, queries):

    for qri, query in enumerate(queries):

        with tag('div', klass="query text"):

            with tag('h4'):

                text(query["text"])

        with tag('div', klass="query comment"):

            with tag('p'):

                text(query["comment"])

        with tag('div', klass="query snippets", style="background-color: #E8E9EB;color:#313638;"):

            for sni, snippet in enumerate(query["snippets"]):

                with tag('a', href=snippet["url"], style="margin-bottom:1em;text-decoration:none;"):

                    text(snippet["title"])

                with tag('p', klass="snippet"):

                    render_text(tag, text, snippet["text"])



def render_questions(tag, text, questions):

    for qi, question in enumerate(questions):

        with tag('div', klass="question entry"):

            with tag('div', klass="question text collapsible"):

                with tag('h3'):

                    text(question["text"])

            with tag('div', klass="queries group"):

                render_queries(tag, text, question["queries"])



def render_tasks(tag, text, tasks):

    for ti, task in enumerate(tasks):

        with tag('div', klass="task entry"):

            with tag('div', klass="task text collapsible"):

                with tag('h2'):

                    text(task["text"])

            with tag('div', klass="questions group"):

                render_questions(tag, text, task["questions"])
from IPython.core.display import display, HTML



styles = ''

with open('../input/styles/snippet_styles.css', 'r') as fin:

    for line in fin:

        styles += line



script = """

    var coll = document.getElementsByClassName("collapsible");

    var i;



    for (i = 0; i < coll.length; i++) {

      coll[i].addEventListener("click", function() {

        this.classList.toggle("active");

        var content = this.nextElementSibling;

        if (content.style.display === "block") {

          content.style.display = "none";

        } else {

          content.style.display = "block";

        }

      });

    }

"""



doc, tag, text = Doc().tagtext()



doc.asis('<!DOCTYPE html>')



with tag('html', lang="en"):

    with tag('head'):

        with doc.tag('style', type='text/css', scoped="scoped"):

            doc.asis(styles)

    with tag('body'):

        with tag('div', klass="tasks group box"):

            render_tasks(tag, text, tasks)

        with tag('script'):

            doc.asis(script)



document = doc.getvalue()



display(HTML(document))
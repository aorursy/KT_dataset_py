!pip install requests-html
from requests_html import HTMLSession

import pycountry as pc

import pandas as pd

import bisect

import numpy as np

import matplotlib.pyplot as plt
url = "https://www.worldometers.info/coronavirus/#countries"

session = HTMLSession()

# web response stored as variable, 'r'

r = session.get(url).html

html_data = r.find('#nav-today', first = True).text.split()
cntr = []

for country in pc.countries:

    cntr.append(country.alpha_2)

    cntr.append(country.name)

    cntr.append("USA")
where_is = {}

for i in cntr:

    if cntr.index(i) % 2 == 0:

            value = cntr.index(i) + 1

            if cntr[value] in html_data:

                where_is[i.lower()] = html_data.index(cntr[value])

                where_is[cntr[value].lower()] = html_data.index(cntr[value])
#this is will be important to create the development

user_choice = 'brazil'

choice = user_choice.lower()

index_graph = where_is[choice]

total_cases = html_data[index_graph + 1]

total_cases = float(total_cases.replace(",", ""))

total_deaths = html_data[index_graph + 3]

total_deaths = float(total_deaths.replace(",",""))

total_recovered = html_data[index_graph + 5]

total_recovered = float(total_recovered.replace(",",""))

serious_cases = html_data[index_graph + 7]

serious_cases = float(serious_cases.replace(",",""))



#tidy data

category_names = ['total cases', 'total deaths', 'serious cases', 'total recovered']

results = {choice:[total_cases, total_deaths, serious_cases, total_recovered]}

print(total_cases)



#create plot

def plots(results, category_names):

    labels = list(results.keys())

    data = np.array(list(results.values()))

    data_cum = data.cumsum(axis=1)

    category_colors = plt.get_cmap('Pastel2')(

        np.linspace(0.15, 0.85, data.shape[1]))



    fig, ax = plt.subplots(figsize=(30, 7))

    ax.invert_yaxis()

    ax.xaxis.set_visible(False)

    ax.set_xlim(0, np.sum(data, axis=1).max())



    for i, (colname, color) in enumerate(zip(category_names, category_colors)):

        widths = data[:, i]

        starts = data_cum[:, i] - widths

        ax.barh(labels, widths, left=starts, height=0.5,

                label=colname, color=color)

        xcenters = starts + widths / 2



        r, g, b, _ = color

        text_color = 'white' if r * g * b < 0.5 else 'black'

        for y, (x, c) in enumerate(zip(xcenters, widths)):

            ax.text(x, y, str(int(c)), ha='center', va='center',

                    color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),

              loc='lower left', fontsize='small')



    return plt, ax





plots(results, category_names)

plt.show()



def percentage(whole):

  return (whole * 100)/total_cases



deaths_p = percentage(total_deaths)

serious_p = percentage(serious_cases)

recovered_p = percentage(total_recovered)



percentage_data = [{'percentage of deaths':deaths_p, 'percentage of serious cases':serious_p,'percentage of recovered cases':recovered_p}]

percentage_df = pd.DataFrame(percentage_data)

percentage_df
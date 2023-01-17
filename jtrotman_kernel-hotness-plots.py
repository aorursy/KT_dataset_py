import pandas as pd

import numpy as np

import os, re, sys

from IPython.core.display import HTML, Image

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
IN_DIR = os.path.join("..", "input", "kernel-hotness-20200222")
def row_text(r):

    m = mmap.get(r.medal, "")

    return (

        f"{m}{r.title}"

        f"<br>"

        f"{r.author_displayName}"

        f"<br>"

        f"Version: {r.versionNumber}"

        f"<br>"

        f"Views: {r.totalViews}"

        f" Votes: {r.totalVotes}"

        f" Ratio: {r.voteRatio:.3f}"

    )



mmap = dict(zip([ "gold", "silver", "bronze" ], "🥇🥈🥉"))



def show(slug):

    title = (' ' + slug).replace('-', ' ')

    title = re.sub(' (\w)', lambda m: f' {m.group(1).upper()}', title).strip()

    df = pd.read_csv(f"{IN_DIR}/{slug}.csv")

    df = df.assign(voteRatio=df.eval("totalViews/(totalVotes+1)"))

    text = df.apply(row_text, 1)

    dates = pd.to_datetime(df.lastRunTime)

    

    display(HTML(

        f"<h1 id='{slug}'>{title}</h1>"

        "<br/>"

        f"{df.shape[0]} Kernels [ <a href='https://www.kaggle.com/c/{slug}/kernels'>Live listing</a> ]"

        "<br/>"

        f"Earliest run: {dates.min().strftime('%c')}"

        "<br/>"

        f"Latest run: {dates.max().strftime('%c')}"

    ))

    

    trace1 = go.Scatter(

        x=df.hotness,

        y=df.totalViews,

        name="Views",

        text=text

    )

    trace2 = go.Scatter(

        x=df.hotness,

        y=df.totalVotes,

        name="Votes",

        text=text,

        yaxis="y2"

    )

    colors = ["#f33", "#33f"]

    data = [trace1, trace2]

    layout = go.Layout(

        title=f"{title} Kernel Votes and Views",

        colorway=colors,

        height=700,

        xaxis=dict(

            title="Hotness"

        ),

        yaxis=dict(

            title="Views",

            titlefont=dict(

                color=colors[0],

            ),

            tickfont=dict(

                color=colors[0],

            ),

        ),

        yaxis2=dict(

            title="Votes",

            type="log",

            titlefont=dict(

                color=colors[1],

            ),

            tickfont=dict(

                color=colors[1],

            ),

            overlaying="y",

            side="right"

        )

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename=slug)
show("ultrasound-nerve-segmentation")
show("shelter-animal-outcomes")
show("noaa-fisheries-steller-sea-lion-population-count")
show("new-york-city-taxi-fare-prediction")
show("kobe-bryant-shot-selection")
show("severstal-steel-defect-detection")
show("instant-gratification")
show("santander-value-prediction-challenge")
show("springleaf-marketing-response")
show("santas-uncertain-bags")
show("otto-group-product-classification-challenge")
show("nfl-big-data-bowl-2020")
show("elo-merchant-category-recommendation")
show("transfer-learning-on-stack-exchange-tags")
show("introducing-kaggle-scripts")
show("bnp-paribas-cardif-claims-management")
show("predict-west-nile-virus")
show("champs-scalar-coupling")
show("home-depot-product-search-relevance")
show("aerial-cactus-identification")
show("ghouls-goblins-and-ghosts-boo")
show("petfinder-adoption-prediction")
show("youtube8m")
show("costa-rican-household-poverty-prediction")
show("prudential-life-insurance-assessment")
show("Kannada-MNIST")
show("LANL-Earthquake-Prediction")
show("melbourne-university-seizure-prediction")
show("planet-understanding-the-amazon-from-space")
show("airbnb-recruiting-new-user-bookings")
show("dog-breed-identification")
show("ashrae-energy-prediction")
show("liberty-mutual-group-property-inspection-prediction")
show("intel-mobileodt-cervical-cancer-screening")
show("talkingdata-adtracking-fraud-detection")
show("two-sigma-financial-news")
show("nyc-taxi-trip-duration")
show("data-science-bowl-2019")
show("jigsaw-unintended-bias-in-toxicity-classification")
show("bike-sharing-demand")
show("ga-customer-revenue-prediction")
show("porto-seguro-safe-driver-prediction")
show("dstl-satellite-imagery-feature-detection")
show("aptos2019-blindness-detection")
show("facebook-v-predicting-check-ins")
show("mercari-price-suggestion-challenge")
show("expedia-hotel-recommendations")
show("mercedes-benz-greener-manufacturing")
show("sf-crime")
show("grupo-bimbo-inventory-demand")
show("ieee-fraud-detection")
show("pubg-finish-placement-prediction")
show("rossmann-store-sales")
show("home-credit-default-risk")
show("jigsaw-toxic-comment-classification-challenge")
show("instacart-market-basket-analysis")
show("zillow-prize-1")
show("outbrain-click-prediction")
show("santander-customer-transaction-prediction")
show("bosch-production-line-performance")
show("predicting-red-hat-business-value")
show("allstate-claims-severity")
show("sberbank-russian-housing-market")
show("leaf-classification")
show("santander-customer-satisfaction")
show("quora-insincere-questions-classification")
show("the-nature-conservancy-fisheries-monitoring")
show("talkingdata-mobile-user-demographics")
show("santander-product-recommendation")
show("dogs-vs-cats-redux-kernels-edition")
show("two-sigma-connect-rental-listing-inquiries")
show("quora-question-pairs")
show("data-science-bowl-2017")
import requests

import re

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

from selenium import webdriver

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.by import By
html1 ='''<div class="col-sm-9 col-md-9 col-xs-12 book-listing">

                <section class="head">

                    <div class="col-sm-8 col-md-8 col-xs-8">

                        <h1> الكتب الأكثر مبيعاً 2019</h1>

                    </div>

                </section>

                                                                            <section class="list">

        <div class="hidden-xs">

            <div class="txt-center">

                    <ul class="pagination">

                                                                                    <li class="active">

                    <span>

                    1                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

        </div>

        </div>

                        <div class="row">

                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'Shusmo \u0634\u0633\u0645\u0648','id':'37291074','price':'15.00','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null,'list':'Product List','position':1}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37291074/s/1035442/category/6220/" title="Shusmo شسمو" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1035442_170x170.jpg" width="135" height="185" alt="Shusmo شسمو">

                                    <h4>Shusmo شسمو</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$15.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $15.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'Shusmo \u0634\u0633\u0645\u0648','id':'37291074','price':'15.00','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/37291074/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0642\u0648\u0642\u0639\u0629 : \u064a\u0648\u0645\u064a\u0627\u062a \u0645\u062a\u0644\u0635\u0635','id':'2784','price':'11.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null,'list':'Product List','position':2}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/1004376.html" title="القوقعة : يوميات متلصص" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1004376_170x170.jpg" width="135" height="185" alt="القوقعة : يوميات متلصص">

                                    <h4>القوقعة : يوميات متلصص</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $11.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0642\u0648\u0642\u0639\u0629 : \u064a\u0648\u0645\u064a\u0627\u062a \u0645\u062a\u0644\u0635\u0635','id':'2784','price':'11.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/2784/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0642\u0648\u0629 \u0639\u0642\u0644\u0643 \u0627\u0644\u0628\u0627\u0637\u0646','id':'36614676','price':'14.00','category':'Literature-fiction \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':3}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36614676/s/5000449/category/6220/" title="قوة عقلك الباطن" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000449_170x170.jpg" width="135" height="185" alt="قوة عقلك الباطن">

                                    <h4>قوة عقلك الباطن</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$14.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $14.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0642\u0648\u0629 \u0639\u0642\u0644\u0643 \u0627\u0644\u0628\u0627\u0637\u0646','id':'36614676','price':'14.00','category':'Literature-fiction \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36614676/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0639 \u0627\u0644\u0646\u0628\u064a \u0635\u0644\u0649 \u0627\u0644\u0644\u0647 \u0639\u0644\u064a\u0647 \u0648\u0633\u0644\u0645','id':'36862809','price':'12.00','category':'Literature-fiction \/ Islamic-books \/ Books','brand':'\u062f\u0627\u0631 \u0643\u0644\u0645\u0627\u062a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0627\u0644\u0643\u0648\u064a\u062a','variant':null,'list':'Product List','position':4}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36862809/s/4101324/category/6220/" title="مع النبي صلى الله عليه وسلم" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4101324_170x170.jpg" width="135" height="185" alt="مع النبي صلى الله عليه وسلم">

                                    <h4>مع النبي صلى الله عليه وسلم</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u0639 \u0627\u0644\u0646\u0628\u064a \u0635\u0644\u0649 \u0627\u0644\u0644\u0647 \u0639\u0644\u064a\u0647 \u0648\u0633\u0644\u0645','id':'36862809','price':'12.00','category':'Literature-fiction \/ Islamic-books \/ Books','quantity':1,'brand':'\u062f\u0627\u0631 \u0643\u0644\u0645\u0627\u062a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0627\u0644\u0643\u0648\u064a\u062a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36862809/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u0644\u0633\u0644\u0629 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631 : \u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634, \u0635\u062f\u0627\u0645 \u0627\u0644\u0645\u0644\u0648\u0643, \u0639\u0627\u0635\u0641\u0629 \u0627\u0644\u0633\u064a\u0648\u0641','id':'36898677','price':'97.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':5}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36898677/s/7000043/category/6220/" title="سلسلة أغنية الجليد والنار : لعبة العروش, صدام الملوك, عاصفة السيوف" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/7000043_170x170.jpg" width="135" height="185" alt="سلسلة أغنية الجليد والنار : لعبة العروش, صدام الملوك, عاصفة السيوف">

                                    <h4>سلسلة أغنية الجليد والنار : لعبة العروش, صدام الملوك, عاصفة السيوف</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$97.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $97.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u0644\u0633\u0644\u0629 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631 : \u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634, \u0635\u062f\u0627\u0645 \u0627\u0644\u0645\u0644\u0648\u0643, \u0639\u0627\u0635\u0641\u0629 \u0627\u0644\u0633\u064a\u0648\u0641','id':'36898677','price':'97.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36898677/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0623\u0628 \u0627\u0644\u063a\u0646\u064a \u0627\u0644\u0623\u0628 \u0627\u0644\u0641\u0642\u064a\u0631','id':'8400','price':'14.98','category':'Business \/ Self-help \/ Books','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':6}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/8400/s/1014741/category/6220/" title="الأب الغني الأب الفقير" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1014741_170x170.jpg" width="135" height="185" alt="الأب الغني الأب الفقير">

                                    <h4>الأب الغني الأب الفقير</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$14.98</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $14.98                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0623\u0628 \u0627\u0644\u063a\u0646\u064a \u0627\u0644\u0623\u0628 \u0627\u0644\u0641\u0642\u064a\u0631','id':'8400','price':'14.98','category':'Business \/ Self-help \/ Books','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/8400/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0633\u0645\u0627\u062d \u0628\u0627\u0644\u0631\u062d\u064a\u0644 : \u0627\u0644\u0637\u0631\u064a\u0642 \u0646\u062d\u0648 \u0627\u0644\u062a\u0633\u0644\u064a\u0645','id':'36750290','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Guidance','brand':'\u062f\u0627\u0631 \u0627\u0644\u062e\u064a\u0627\u0644 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':7}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36750290/s/3094005/category/6220/" title="السماح بالرحيل : الطريق نحو التسليم" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3094005_170x170.jpg" width="135" height="185" alt="السماح بالرحيل : الطريق نحو التسليم">

                                    <h4>السماح بالرحيل : الطريق نحو التسليم</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$13.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $13.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0633\u0645\u0627\u062d \u0628\u0627\u0644\u0631\u062d\u064a\u0644 : \u0627\u0644\u0637\u0631\u064a\u0642 \u0646\u062d\u0648 \u0627\u0644\u062a\u0633\u0644\u064a\u0645','id':'36750290','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Guidance','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062e\u064a\u0627\u0644 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36750290/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0641\u0646 \u0627\u0644\u0644\u0627\u0645\u0628\u0627\u0644\u0627\u0629 : \u0644\u0639\u064a\u0634 \u062d\u064a\u0627\u0629 \u062a\u062e\u0627\u0644\u0641 \u0627\u0644\u0645\u0623\u0644\u0648\u0641','id':'36974116','price':'10.00','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u00a0\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':8}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36974116/s/3106057/category/6220/" title="فن اللامبالاة : لعيش حياة تخالف المألوف" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3106057_170x170.jpg" width="135" height="185" alt="فن اللامبالاة : لعيش حياة تخالف المألوف">

                                    <h4>فن اللامبالاة : لعيش حياة تخالف المألوف</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0641\u0646 \u0627\u0644\u0644\u0627\u0645\u0628\u0627\u0644\u0627\u0629 : \u0644\u0639\u064a\u0634 \u062d\u064a\u0627\u0629 \u062a\u062e\u0627\u0644\u0641 \u0627\u0644\u0645\u0623\u0644\u0648\u0641','id':'36974116','price':'10.00','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u00a0\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36974116/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0639\u0635\u0641\u0648\u0631\u064a\u0629','id':'309966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':9}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/309966/s/3026103/category/6220/" title="العصفورية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3026103_170x170.jpg" width="135" height="185" alt="العصفورية">

                                    <h4>العصفورية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0639\u0635\u0641\u0648\u0631\u064a\u0629','id':'309966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/309966/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0627 \u062a\u0642\u0648\u0644\u064a \u0625\u0646\u0643 \u062e\u0627\u0626\u0641\u0629','id':'36772107','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0645\u062a\u0648\u0633\u0637','variant':null,'list':'Product List','position':10}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36772107/s/3095205/category/6220/" title="لا تقولي إنك خائفة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3095205_170x170.jpg" width="135" height="185" alt="لا تقولي إنك خائفة">

                                    <h4>لا تقولي إنك خائفة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$15.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $15.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0627 \u062a\u0642\u0648\u0644\u064a \u0625\u0646\u0643 \u062e\u0627\u0626\u0641\u0629','id':'36772107','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0645\u062a\u0648\u0633\u0637','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36772107/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0635\u0627\u062d\u0628 \u0627\u0644\u0638\u0644 \u0627\u0644\u0637\u0648\u064a\u0644','id':'37036122','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0631\u0627\u0641\u062f\u064a\u0646 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':11}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37036122/s/3107490/category/6220/" title="صاحب الظل الطويل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107490_170x170.jpg" width="135" height="185" alt="صاحب الظل الطويل">

                                    <h4>صاحب الظل الطويل</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0635\u0627\u062d\u0628 \u0627\u0644\u0638\u0644 \u0627\u0644\u0637\u0648\u064a\u0644','id':'37036122','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0631\u0627\u0641\u062f\u064a\u0646 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/37036122/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u064a\u062f\u0627\u062a \u0627\u0644\u0642\u0645\u0631','id':'377164','price':'9.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null,'list':'Product List','position':12}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/377164/s/3075604/category/6220/" title="سيدات القمر" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3075604_170x170.jpg" width="135" height="185" alt="سيدات القمر">

                                    <h4>سيدات القمر</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u064a\u062f\u0627\u062a \u0627\u0644\u0642\u0645\u0631','id':'377164','price':'9.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/377164/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u062d\u0627\u0631\u0633 \u0641\u064a \u062d\u0642\u0644 \u0627\u0644\u0634\u0648\u0641\u0627\u0646','id':'36789333','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u062f\u0649 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':13}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36789333/s/8000592/category/6220/" title="الحارس في حقل الشوفان" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/8000592_170x170.jpg" width="135" height="185" alt="الحارس في حقل الشوفان">

                                    <h4>الحارس في حقل الشوفان</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u062d\u0627\u0631\u0633 \u0641\u064a \u062d\u0642\u0644 \u0627\u0644\u0634\u0648\u0641\u0627\u0646','id':'36789333','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u062f\u0649 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36789333/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0642\u0635\u062a\u064a : \u062e\u0645\u0633\u064a\u0646 \u0642\u0635\u0629 \u0641\u064a \u062e\u0645\u0633\u064a\u0646 \u0639\u0627\u0645\u0627\u064b','id':'37134943','price':'24.23','category':'Literature-fiction \/ Books \/ Biography','brand':'Explorer Publishing &amp; Distribution LLC','variant':null,'list':'Product List','position':14}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37134943/s/4203056/category/6220/" title="قصتي : خمسين قصة في خمسين عاماً" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4203056_170x170.jpg" width="135" height="185" alt="قصتي : خمسين قصة في خمسين عاماً">

                                    <h4>قصتي : خمسين قصة في خمسين عاماً</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$24.23</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $24.23                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0642\u0635\u062a\u064a : \u062e\u0645\u0633\u064a\u0646 \u0642\u0635\u0629 \u0641\u064a \u062e\u0645\u0633\u064a\u0646 \u0639\u0627\u0645\u0627\u064b','id':'37134943','price':'24.23','category':'Literature-fiction \/ Books \/ Biography','quantity':1,'brand':'Explorer Publishing &amp; Distribution LLC','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/37134943/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0647\u0632\u0644\u0629 \u0627\u0644\u0639\u0642\u0644 \u0627\u0644\u0628\u0634\u0631\u064a','id':'419880','price':'13.00','category':'Islamic-books \/ Books \/ History-civilization','brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null,'list':'Product List','position':15}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3016261.html" title="مهزلة العقل البشري" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3016261_170x170.jpg" width="135" height="185" alt="مهزلة العقل البشري">

                                    <h4>مهزلة العقل البشري</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$13.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $13.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u0647\u0632\u0644\u0629 \u0627\u0644\u0639\u0642\u0644 \u0627\u0644\u0628\u0634\u0631\u064a','id':'419880','price':'13.00','category':'Islamic-books \/ Books \/ History-civilization','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/419880/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'Pharma Guide Pre-Work : Pharmacy Practice for Postgraduate_ Third Edition','id':'37156346','price':'20.00','category':'Enbooks \/ Best-seller-books-2019','brand':'University Book Center','variant':null,'list':'Product List','position':16}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37156346/s/9789779056098/category/6220/" title="Pharma Guide Pre-Work : Pharmacy Practice for Postgraduate_ Third Edition" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9789779056098_170x170.jpg" width="135" height="185" alt="Pharma Guide Pre-Work : Pharmacy Practice for Postgraduate_ Third Edition">

                                    <h4>Pharma Guide Pre-Work : Pharmacy Practice for Postgraduate_ Third Edition</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$20.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $20.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'Pharma Guide Pre-Work : Pharmacy Practice for Postgraduate_ Third Edition','id':'37156346','price':'20.00','category':'Enbooks \/ Best-seller-books-2019','quantity':1,'brand':'University Book Center','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/37156346/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0642\u0648\u0627\u0639\u062f \u0627\u0644\u0639\u0634\u0642 \u0627\u0644\u0623\u0631\u0628\u0639\u0648\u0646','id':'33792825','price':'10.50','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null,'list':'Product List','position':17}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                30%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3087504.html" title="قواعد العشق الأربعون" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3087504_170x170.jpg" width="135" height="185" alt="قواعد العشق الأربعون">

                                    <h4>قواعد العشق الأربعون</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$15.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $15.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$10.50</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0642\u0648\u0627\u0639\u062f \u0627\u0644\u0639\u0634\u0642 \u0627\u0644\u0623\u0631\u0628\u0639\u0648\u0646','id':'33792825','price':'10.50','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/33792825/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u063a\u062f\u0627\u064b \u0623\u062c\u0645\u0644','id':'36830969','price':'10.50','category':'Literature-fiction \/ Self-help \/ Books','brand':'\u062f\u0627\u0631 \u0645\u062f\u0627\u0631\u0643 \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':18}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36830969/s/5000960/category/6220/" title="غداً أجمل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000960_170x170.jpg" width="135" height="185" alt="غداً أجمل">

                                    <h4>غداً أجمل</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.50                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u063a\u062f\u0627\u064b \u0623\u062c\u0645\u0644','id':'36830969','price':'10.50','category':'Literature-fiction \/ Self-help \/ Books','quantity':1,'brand':'\u062f\u0627\u0631 \u0645\u062f\u0627\u0631\u0643 \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36830969/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062e\u0631\u0627\u0628 : \u0643\u062a\u0627\u0628 \u0639\u0646 \u0627\u0644\u0623\u0645\u0644','id':'37233233','price':'11.40','category':'Books \/ Social-sciences-humanities \/ Sociology','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':19}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37233233/s/3119444/category/6220/" title="خراب : كتاب عن الأمل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3119444_170x170.jpg" width="135" height="185" alt="خراب : كتاب عن الأمل">

                                    <h4>خراب : كتاب عن الأمل</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.40</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $12.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$11.40</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062e\u0631\u0627\u0628 : \u0643\u062a\u0627\u0628 \u0639\u0646 \u0627\u0644\u0623\u0645\u0644','id':'37233233','price':'11.40','category':'Books \/ Social-sciences-humanities \/ Sociology','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/37233233/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0642\u0648\u0645 \u0642\u064a\u0644\u0627','id':'36515275','price':'14.00','category':'Literature-fiction \/ Books \/ Philosophy','brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':20}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3090408.html" title="أقوم قيلا" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3090408_170x170.jpg" width="135" height="185" alt="أقوم قيلا">

                                    <h4>أقوم قيلا</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$14.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $14.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0642\u0648\u0645 \u0642\u064a\u0644\u0627','id':'36515275','price':'14.00','category':'Literature-fiction \/ Books \/ Philosophy','quantity':1,'brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P3A9MSZfX19mcm9tX3N0b3JlPWVu/product/36515275/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                               <!-- End of ITEM -->

        <div class="txt-center">

                                <ul class="pagination">

                                                                                    <li class="active">

                    <span>

                    1                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

                </div>

    </div></section>





                            </div> '''
html2 ='''<div class="col-sm-9 col-md-9 col-xs-12 book-listing">

                <section class="head">

                    <div class="col-sm-8 col-md-8 col-xs-8">

                        <h1> الكتب الأكثر مبيعاً 2019</h1>

                    </div>

                </section>

                                                                            <section class="list">

        <div class="hidden-xs">

            <div class="txt-center">

                    <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    2                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

        </div>

        </div>

                        <div class="row">

                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062e\u0631\u0627\u0628 : \u0643\u062a\u0627\u0628 \u0639\u0646 \u0627\u0644\u0623\u0645\u0644','id':'37233233','price':'11.40','category':'Books \/ Social-sciences-humanities \/ Sociology','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':1}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37233233/s/3119444/category/6220/" title="خراب : كتاب عن الأمل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3119444_170x170.jpg" width="135" height="185" alt="خراب : كتاب عن الأمل">

                                    <h4>خراب : كتاب عن الأمل</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.40</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $12.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$11.40</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062e\u0631\u0627\u0628 : \u0643\u062a\u0627\u0628 \u0639\u0646 \u0627\u0644\u0623\u0645\u0644','id':'37233233','price':'11.40','category':'Books \/ Social-sciences-humanities \/ Sociology','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u0631\u0645\u0644 - \u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37233233/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062d\u064a\u0648\u0646\u0629 \u0627\u0644\u0625\u0646\u0633\u0627\u0646','id':'4336','price':'10.00','category':'Literature-fiction \/ General \/ Top-selling-books-for-2016','brand':'\u062f\u0627\u0631 \u0645\u0645\u062f\u0648\u062d \u0639\u062f\u0648\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':2}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/4336/s/1007788/category/6220/" title="حيونة الإنسان" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1007788_170x170.jpg" width="135" height="185" alt="حيونة الإنسان">

                                    <h4>حيونة الإنسان</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062d\u064a\u0648\u0646\u0629 \u0627\u0644\u0625\u0646\u0633\u0627\u0646','id':'4336','price':'10.00','category':'Literature-fiction \/ General \/ Top-selling-books-for-2016','quantity':1,'brand':'\u062f\u0627\u0631 \u0645\u0645\u062f\u0648\u062d \u0639\u062f\u0648\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/4336/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0628\u0624\u0633\u0627\u0621','id':'36627882','price':'9.50','category':'Literature-fiction \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0641\u0646\u0648\u0646 \u0648\u0627\u0644\u0622\u062f\u0627\u0628','variant':null,'list':'Product List','position':3}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36627882/s/1023717/category/6220/" title="البؤساء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1023717_170x170.jpg" width="135" height="185" alt="البؤساء">

                                    <h4>البؤساء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.50                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0628\u0624\u0633\u0627\u0621','id':'36627882','price':'9.50','category':'Literature-fiction \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0641\u0646\u0648\u0646 \u0648\u0627\u0644\u0622\u062f\u0627\u0628','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36627882/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u0644\u0627\u0645\u064c \u0639\u0644\u0649 \u0625\u0628\u0631\u0627\u0647\u064a\u0645','id':'36883430','price':'7.00','category':'Literature-fiction \/ Books \/ Stories','brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0627\u0631\u0641 \u0627\u0644\u0625\u0633\u0644\u0627\u0645\u064a\u0629 \u0627\u0644\u062b\u0642\u0627\u0641\u064a\u0629','variant':null,'list':'Product List','position':4}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36883430/s/3103818/category/6220/" title="سلامٌ على إبراهيم" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3103818_170x170.jpg" width="135" height="185" alt="سلامٌ على إبراهيم">

                                    <h4>سلامٌ على إبراهيم</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u0644\u0627\u0645\u064c \u0639\u0644\u0649 \u0625\u0628\u0631\u0627\u0647\u064a\u0645','id':'36883430','price':'7.00','category':'Literature-fiction \/ Books \/ Stories','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0627\u0631\u0641 \u0627\u0644\u0625\u0633\u0644\u0627\u0645\u064a\u0629 \u0627\u0644\u062b\u0642\u0627\u0641\u064a\u0629','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36883430/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u062a\u0627\u0647\u0629 \u0627\u0644\u0623\u0631\u0648\u0627\u062d','id':'37352764','price':'19.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null,'list':'Product List','position':5}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37352764/s/3121394/category/6220/" title="متاهة الأرواح" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3121394_170x170.jpg" width="135" height="185" alt="متاهة الأرواح">

                                    <h4>متاهة الأرواح</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$20.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$19.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $20.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$19.00</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u062a\u0627\u0647\u0629 \u0627\u0644\u0623\u0631\u0648\u0627\u062d','id':'37352764','price':'19.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37352764/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u062e\u064a\u0645\u064a\u0627\u0626\u064a','id':'298856','price':'4.80','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':6}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                40%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3033435.html" title="الخيميائي" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3033435_170x170.jpg" width="135" height="185" alt="الخيميائي">

                                    <h4>الخيميائي</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$8.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$4.80</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $8.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$4.80</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u062e\u064a\u0645\u064a\u0627\u0626\u064a','id':'298856','price':'4.80','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/298856/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062c\u0648\u0631\u062c \u0623\u0648\u0631\u0648\u064a\u0644 : 1984','id':'36770337','price':'6.50','category':'Literature-fiction \/ Politics \/ Books','brand':'\u0627\u0644\u0623\u0647\u0644\u064a\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':7}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                50%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36770337/s/1024519/category/6220/" title="جورج أورويل : 1984" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1024519_170x170.jpg" width="135" height="185" alt="جورج أورويل : 1984">

                                    <h4>جورج أورويل : 1984</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$13.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $13.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$6.50</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062c\u0648\u0631\u062c \u0623\u0648\u0631\u0648\u064a\u0644 : 1984','id':'36770337','price':'6.50','category':'Literature-fiction \/ Politics \/ Books','quantity':1,'brand':'\u0627\u0644\u0623\u0647\u0644\u064a\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36770337/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0642\u0644\u0642 \u0627\u0644\u0633\u0639\u064a \u0625\u0644\u0649 \u0627\u0644\u0645\u0643\u0627\u0646\u0629','id':'36951751','price':'9.00','category':'Books \/ Philosophy \/ Best-seller-books-2019','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':8}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36951751/s/3105405/category/6220/" title="قلق السعي إلى المكانة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3105405_170x170.jpg" width="135" height="185" alt="قلق السعي إلى المكانة">

                                    <h4>قلق السعي إلى المكانة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0642\u0644\u0642 \u0627\u0644\u0633\u0639\u064a \u0625\u0644\u0649 \u0627\u0644\u0645\u0643\u0627\u0646\u0629','id':'36951751','price':'9.00','category':'Books \/ Philosophy \/ Best-seller-books-2019','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36951751/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0631\u0633\u0627\u0626\u0644 \u063a\u0633\u0627\u0646 \u0643\u0646\u0641\u0627\u0646\u064a \u0625\u0644\u0649 \u063a\u0627\u062f\u0629 \u0627\u0644\u0633\u0645\u0627\u0646','id':'371102','price':'8.50','category':'Literature-fiction \/ Literary-texts \/ Top1oocat','brand':'\u062f\u0627\u0631 \u0627\u0644\u0637\u0644\u064a\u0639\u0629 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':9}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3018400.html" title="رسائل غسان كنفاني إلى غادة السمان" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3018400_170x170.jpg" width="135" height="185" alt="رسائل غسان كنفاني إلى غادة السمان">

                                    <h4>رسائل غسان كنفاني إلى غادة السمان</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$8.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $8.50                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0631\u0633\u0627\u0626\u0644 \u063a\u0633\u0627\u0646 \u0643\u0646\u0641\u0627\u0646\u064a \u0625\u0644\u0649 \u063a\u0627\u062f\u0629 \u0627\u0644\u0633\u0645\u0627\u0646','id':'371102','price':'8.50','category':'Literature-fiction \/ Literary-texts \/ Top1oocat','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0637\u0644\u064a\u0639\u0629 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/371102/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0644\u064a\u0627\u0644\u064a \u0627\u0644\u0628\u064a\u0636\u0627\u0621','id':'36783231','price':'6.00','category':'Literature-fiction \/ Politics \/ Books','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null,'list':'Product List','position':10}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36783231/s/3096522/category/6220/" title="الليالي البيضاء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3096522_170x170.jpg" width="135" height="185" alt="الليالي البيضاء">

                                    <h4>الليالي البيضاء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $6.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0644\u064a\u0627\u0644\u064a \u0627\u0644\u0628\u064a\u0636\u0627\u0621','id':'36783231','price':'6.00','category':'Literature-fiction \/ Politics \/ Books','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36783231/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u0647\u064a\u0627\u064b \u0643\u0641\u0631\u0627\u0642','id':'37026711','price':'12.00','category':'Literature-fiction \/ Books \/ Stories','brand':'\u0646\u0648\u0641\u0644 \/ \u0647\u0627\u0634\u064a\u062a \u0623\u0646\u0637\u0648\u0627\u0646','variant':null,'list':'Product List','position':11}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37026711/s/3107338/category/6220/" title="شهياً كفراق" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107338_170x170.jpg" width="135" height="185" alt="شهياً كفراق">

                                    <h4>شهياً كفراق</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u0647\u064a\u0627\u064b \u0643\u0641\u0631\u0627\u0642','id':'37026711','price':'12.00','category':'Literature-fiction \/ Books \/ Stories','quantity':1,'brand':'\u0646\u0648\u0641\u0644 \/ \u0647\u0627\u0634\u064a\u062a \u0623\u0646\u0637\u0648\u0627\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37026711/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0643\u0627\u0641\u0643\u0627 \u0639\u0644\u0649 \u0627\u0644\u0634\u0627\u0637\u0626','id':'397692','price':'16.00','category':'Literature-fiction \/ Novels \/ Top1oocat','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null,'list':'Product List','position':12}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3070376.html" title="كافكا على الشاطئ" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3070376_170x170.jpg" width="135" height="185" alt="كافكا على الشاطئ">

                                    <h4>كافكا على الشاطئ</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$16.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $16.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0643\u0627\u0641\u0643\u0627 \u0639\u0644\u0649 \u0627\u0644\u0634\u0627\u0637\u0626','id':'397692','price':'16.00','category':'Literature-fiction \/ Novels \/ Top1oocat','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/397692/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0646\u0638\u0631\u064a\u0629 \u0627\u0644\u0641\u0633\u062a\u0642 : \u0643\u062a\u0627\u0628 \u0633\u064a\u063a\u064a\u0631 \u0637\u0631\u064a\u0642\u0629 \u062a\u0641\u0643\u064a\u0631\u0643 \u0648\u062d\u0643\u0645\u0643 \u0639\u0644\u0649 \u0627\u0644\u0623\u0634\u064a\u0627\u0621','id':'36796575','price':'9.33','category':'Islamic-books \/ Books \/ Thought','brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null,'list':'Product List','position':13}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36796575/s/5000798/category/6220/" title="نظرية الفستق : كتاب سيغير طريقة تفكيرك وحكمك على الأشياء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000798_170x170.jpg" width="135" height="185" alt="نظرية الفستق : كتاب سيغير طريقة تفكيرك وحكمك على الأشياء">

                                    <h4>نظرية الفستق : كتاب سيغير طريقة تفكيرك وحكمك على الأشياء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.33</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.33                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0646\u0638\u0631\u064a\u0629 \u0627\u0644\u0641\u0633\u062a\u0642 : \u0643\u062a\u0627\u0628 \u0633\u064a\u063a\u064a\u0631 \u0637\u0631\u064a\u0642\u0629 \u062a\u0641\u0643\u064a\u0631\u0643 \u0648\u062d\u0643\u0645\u0643 \u0639\u0644\u0649 \u0627\u0644\u0623\u0634\u064a\u0627\u0621','id':'36796575','price':'9.33','category':'Islamic-books \/ Books \/ Thought','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36796575/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'40 \u0623\u0631\u0628\u0639\u0648\u0646','id':'37106880','price':'11.20','category':'Books \/ Daily \/ Best-seller-books-2019','brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null,'list':'Product List','position':14}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                30%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37106880/s/3107990/category/6220/" title="40 أربعون" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107990_170x170.jpg" width="135" height="185" alt="40 أربعون">

                                    <h4>40 أربعون</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$16.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.20</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $16.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$11.20</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'40 \u0623\u0631\u0628\u0639\u0648\u0646','id':'37106880','price':'11.20','category':'Books \/ Daily \/ Best-seller-books-2019','quantity':1,'brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37106880/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0639\u062f\u0627\u0621 \u0627\u0644\u0637\u0627\u0626\u0631\u0629 \u0627\u0644\u0648\u0631\u0642\u064a\u0629','id':'18928110','price':'14.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u062c\u0627\u0645\u0639\u0629 \u062d\u0645\u062f \u0628\u0646 \u062e\u0644\u064a\u0641\u0629 \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':15}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/18928110/s/4000003/category/6220/" title="عداء الطائرة الورقية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4000003_170x170.jpg" width="135" height="185" alt="عداء الطائرة الورقية">

                                    <h4>عداء الطائرة الورقية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$14.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $14.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0639\u062f\u0627\u0621 \u0627\u0644\u0637\u0627\u0626\u0631\u0629 \u0627\u0644\u0648\u0631\u0642\u064a\u0629','id':'18928110','price':'14.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u062c\u0627\u0645\u0639\u0629 \u062d\u0645\u062f \u0628\u0646 \u062e\u0644\u064a\u0641\u0629 \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/18928110/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0631\u0636 \u0627\u0644\u0633\u0627\u0641\u0644\u064a\u0646','id':'36828523','price':'9.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':16}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36828523/s/5000953/category/6220/" title="أرض السافلين" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000953_170x170.jpg" width="135" height="185" alt="أرض السافلين">

                                    <h4>أرض السافلين</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0631\u0636 \u0627\u0644\u0633\u0627\u0641\u0644\u064a\u0646','id':'36828523','price':'9.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36828523/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0646 \u064a\u0645\u0648\u062a','id':'37154727','price':'5.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0627\u0644\u0631\u0648\u0627\u0642 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':17}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37154727/s/9009056/category/6220/" title="لن يموت" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9009056_170x170.jpg" width="135" height="185" alt="لن يموت">

                                    <h4>لن يموت</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$5.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $5.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0646 \u064a\u0645\u0648\u062a','id':'37154727','price':'5.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0627\u0644\u0631\u0648\u0627\u0642 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37154727/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062b\u0644\u0627\u062b\u064a\u0629 \u063a\u0631\u0646\u0627\u0637\u0629','id':'36498237','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null,'list':'Product List','position':18}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36498237/s/1020568/category/6220/" title="ثلاثية غرناطة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1020568_170x170.jpg" width="135" height="185" alt="ثلاثية غرناطة">

                                    <h4>ثلاثية غرناطة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$15.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $15.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062b\u0644\u0627\u062b\u064a\u0629 \u063a\u0631\u0646\u0627\u0637\u0629','id':'36498237','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36498237/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'Milk and Honey','id':'36855919','price':'13.96','category':'Enbooks \/ Best-seller-books-2019','brand':'Andrews McMeel Publishing','variant':null,'list':'Product List','position':19}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36855919/s/9781449474256/category/6220/" title="Milk and Honey" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9781449474256_170x170.jpg" width="135" height="185" alt="Milk and Honey">

                                    <h4>Milk and Honey</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$13.96</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $13.96                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'Milk and Honey','id':'36855919','price':'13.96','category':'Enbooks \/ Best-seller-books-2019','quantity':1,'brand':'Andrews McMeel Publishing','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/36855919/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0645\u0631\u062d\u0644\u0629 \u0627\u0644\u0645\u0644\u0643\u064a\u0629','id':'37191906','price':'9.35','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u062c\u0631\u064a\u0633\u064a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0625\u0639\u0644\u0627\u0646 - \u0627\u0644\u0645\u0645\u0644\u0643\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0647 \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null,'list':'Product List','position':20}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37191906/s/5005580/category/6220/" title="المرحلة الملكية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5005580_170x170.jpg" width="135" height="185" alt="المرحلة الملكية">

                                    <h4>المرحلة الملكية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.35</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.35                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0645\u0631\u062d\u0644\u0629 \u0627\u0644\u0645\u0644\u0643\u064a\u0629','id':'37191906','price':'9.35','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u062c\u0631\u064a\u0633\u064a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0625\u0639\u0644\u0627\u0646 - \u0627\u0644\u0645\u0645\u0644\u0643\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0647 \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0y/product/37191906/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                               <!-- End of ITEM -->

        <div class="txt-center">

                                <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    2                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

                </div>

    </div></section>





                            </div> '''
html3 = '''<div class="col-sm-9 col-md-9 col-xs-12 book-listing">

                <section class="head">

                    <div class="col-sm-8 col-md-8 col-xs-8">

                        <h1> الكتب الأكثر مبيعاً 2019</h1>

                    </div>

                </section>

                                                                            <section class="list">

        <div class="hidden-xs">

            <div class="txt-center">

                    <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    3                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

        </div>

        </div>

                        <div class="row">

                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0648\u0644\u0627\u062f \u062d\u0627\u0631\u062a\u0646\u0627','id':'2972','price':'19.60','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null,'list':'Product List','position':1}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/1006424.html" title="أولاد حارتنا" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1006424_170x170.jpg" width="135" height="185" alt="أولاد حارتنا">

                                    <h4>أولاد حارتنا</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$19.60</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $19.60                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0648\u0644\u0627\u062f \u062d\u0627\u0631\u062a\u0646\u0627','id':'2972','price':'19.60','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/2972/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u063a\u0646\u0649 \u0631\u062c\u0644 \u0641\u064a \u0628\u0627\u0628\u0644','id':'36624570','price':'6.09','category':'Politics \/ Self-help \/ Books','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':2}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36624570/s/5000462/category/6220/" title="أغنى رجل في بابل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000462_170x170.jpg" width="135" height="185" alt="أغنى رجل في بابل">

                                    <h4>أغنى رجل في بابل</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.09</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $6.09                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u063a\u0646\u0649 \u0631\u062c\u0644 \u0641\u064a \u0628\u0627\u0628\u0644','id':'36624570','price':'6.09','category':'Politics \/ Self-help \/ Books','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36624570/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u0646\u062c\u062d \u0642\u0628\u0644 \u0623\u0646 \u062a\u0628\u062f\u0623','id':'36869247','price':'10.00','category':'Books \/ Best-seller-books-2019','brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null,'list':'Product List','position':3}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36869247/s/1029818/category/6220/" title="كيف تنجح قبل أن تبدأ" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1029818_170x170.jpg" width="135" height="185" alt="كيف تنجح قبل أن تبدأ">

                                    <h4>كيف تنجح قبل أن تبدأ</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u0646\u062c\u062d \u0642\u0628\u0644 \u0623\u0646 \u062a\u0628\u062f\u0623','id':'36869247','price':'10.00','category':'Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36869247/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0639\u064a\u0634 \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u062d\u0627\u0636\u0631','id':'37297153','price':'6.40','category':'Books \/ Philosophy \/ Best-seller-books-2019','brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':4}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                20%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37297153/s/3120483/category/6220/" title="الذكاء العيش في الوقت الحاضر" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3120483_170x170.jpg" width="135" height="185" alt="الذكاء العيش في الوقت الحاضر">

                                    <h4>الذكاء العيش في الوقت الحاضر</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$8.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.40</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $8.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$6.40</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0639\u064a\u0634 \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u062d\u0627\u0636\u0631','id':'37297153','price':'6.40','category':'Books \/ Philosophy \/ Best-seller-books-2019','quantity':1,'brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37297153/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0631\u0648\u0627\u064a\u0629 1984','id':'268292','price':'10.00','category':'Literature-fiction \/ Novels \/ Top1oocat','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null,'list':'Product List','position':5}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3059915.html" title="رواية 1984" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3059915_170x170.jpg" width="135" height="185" alt="رواية 1984">

                                    <h4>رواية 1984</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0631\u0648\u0627\u064a\u0629 1984','id':'268292','price':'10.00','category':'Literature-fiction \/ Novels \/ Top1oocat','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/268292/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0648\u062a \u0635\u063a\u064a\u0631','id':'36761834','price':'18.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':6}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36761834/s/3094522/category/6220/" title="موت صغير" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3094522_170x170.jpg" width="135" height="185" alt="موت صغير">

                                    <h4>موت صغير</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$18.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $18.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u0648\u062a \u0635\u063a\u064a\u0631','id':'36761834','price':'18.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36761834/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u062c\u0631\u062a\u064a \u0634\u062c\u0631\u0629 \u0627\u0644\u0628\u0631\u062a\u0642\u0627\u0644 \u0627\u0644\u0631\u0627\u0626\u0639\u0629','id':'36922032','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0633\u0643\u064a\u0644\u064a\u0627\u0646\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':7}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36922032/s/1030292/category/6220/" title="شجرتي شجرة البرتقال الرائعة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1030292_170x170.jpg" width="135" height="185" alt="شجرتي شجرة البرتقال الرائعة">

                                    <h4>شجرتي شجرة البرتقال الرائعة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u062c\u0631\u062a\u064a \u0634\u062c\u0631\u0629 \u0627\u0644\u0628\u0631\u062a\u0642\u0627\u0644 \u0627\u0644\u0631\u0627\u0626\u0639\u0629','id':'36922032','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0633\u0643\u064a\u0644\u064a\u0627\u0646\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36922032/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062d\u064a\u0627\u0629 \u0641\u064a \u0627\u0644\u0625\u062f\u0627\u0631\u0629','id':'362692','price':'10.00','category':'Books \/ Biography-memoir \/ Top1oocat','brand':'\u0627\u0644\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u062f\u0631\u0627\u0633\u0627\u062a \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':8}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/362692/s/3019715/category/6220/" title="حياة في الإدارة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3019715_170x170.jpg" width="135" height="185" alt="حياة في الإدارة">

                                    <h4>حياة في الإدارة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062d\u064a\u0627\u0629 \u0641\u064a \u0627\u0644\u0625\u062f\u0627\u0631\u0629','id':'362692','price':'10.00','category':'Books \/ Biography-memoir \/ Top1oocat','quantity':1,'brand':'\u0627\u0644\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u062f\u0631\u0627\u0633\u0627\u062a \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/362692/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0641\u0646 \u0623\u0646 \u062a\u0643\u0648\u0646 \u062f\u0627\u0626\u0645\u0627\u064b \u0639\u0644\u0649 \u0635\u0648\u0627\u0628','id':'36773795','price':'5.00','category':'Books \/ Philosophy \/ Top-selling-books-for-2016','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0636\u0641\u0627\u0641','variant':null,'list':'Product List','position':9}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36773795/s/3095373/category/6220/" title="فن أن تكون دائماً على صواب" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3095373_170x170.jpg" width="135" height="185" alt="فن أن تكون دائماً على صواب">

                                    <h4>فن أن تكون دائماً على صواب</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$5.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $5.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0641\u0646 \u0623\u0646 \u062a\u0643\u0648\u0646 \u062f\u0627\u0626\u0645\u0627\u064b \u0639\u0644\u0649 \u0635\u0648\u0627\u0628','id':'36773795','price':'5.00','category':'Books \/ Philosophy \/ Top-selling-books-for-2016','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0636\u0641\u0627\u0641','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36773795/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0643\u0633\u062a\u0627\u0633\u064a 65 \u064a\u0648\u0645','id':'37013627','price':'29.00','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':10}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37013627/s/5003407/category/6220/" title="اكستاسي 65 يوم" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5003407_170x170.jpg" width="135" height="185" alt="اكستاسي 65 يوم">

                                    <h4>اكستاسي 65 يوم</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$29.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $29.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0643\u0633\u062a\u0627\u0633\u064a 65 \u064a\u0648\u0645','id':'37013627','price':'29.00','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37013627/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u0642\u0629 \u0627\u0644\u062d\u0631\u064a\u0629','id':'379966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0631\u064a\u0627\u0636 \u0627\u0644\u0631\u064a\u0633 \u0644\u0644\u0643\u062a\u0628 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':11}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/379966/s/3004707/category/6220/" title="شقة الحرية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3004707_170x170.jpg" width="135" height="185" alt="شقة الحرية">

                                    <h4>شقة الحرية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u0642\u0629 \u0627\u0644\u062d\u0631\u064a\u0629','id':'379966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0631\u064a\u0627\u0636 \u0627\u0644\u0631\u064a\u0633 \u0644\u0644\u0643\u062a\u0628 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/379966/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u062c\u064a\u0646 \u0627\u0644\u0633\u0645\u0627\u0621','id':'37068467','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null,'list':'Product List','position':12}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37068467/s/3107609/category/6220/" title="سجين السماء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107609_170x170.jpg" width="135" height="185" alt="سجين السماء">

                                    <h4>سجين السماء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u062c\u064a\u0646 \u0627\u0644\u0633\u0645\u0627\u0621','id':'37068467','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37068467/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0648\u0639\u0627\u0638 \u0627\u0644\u0633\u0644\u0627\u0637\u064a\u0646','id':'428642','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Sociology','brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null,'list':'Product List','position':13}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3073341.html" title="وعاظ السلاطين" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3073341_170x170.jpg" width="135" height="185" alt="وعاظ السلاطين">

                                    <h4>وعاظ السلاطين</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$13.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $13.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0648\u0639\u0627\u0638 \u0627\u0644\u0633\u0644\u0627\u0637\u064a\u0646','id':'428642','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Sociology','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/428642/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0646\u0627 \u064a\u0648\u0633\u0641','id':'37135143','price':'7.05','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0645\u0635\u0631','variant':null,'list':'Product List','position':14}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37135143/s/9008662/category/6220/" title="أنا يوسف" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9008662_170x170.jpg" width="135" height="185" alt="أنا يوسف">

                                    <h4>أنا يوسف</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.05</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.05                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0646\u0627 \u064a\u0648\u0633\u0641','id':'37135143','price':'7.05','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0645\u0635\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37135143/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0641\u0649 \u0642\u0644\u0628\u0649 \u0623\u0646\u062b\u0649 \u0639\u0628\u0631\u064a\u0629','id':'34756961','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0643\u064a\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':15}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/1017636.html" title="فى قلبى أنثى عبرية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1017636_170x170.jpg" width="135" height="185" alt="فى قلبى أنثى عبرية">

                                    <h4>فى قلبى أنثى عبرية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0641\u0649 \u0642\u0644\u0628\u0649 \u0623\u0646\u062b\u0649 \u0639\u0628\u0631\u064a\u0629','id':'34756961','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0643\u064a\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/34756961/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0623\u0646\u0643 \u0627\u0644\u0644\u0647 : \u0631\u062d\u0644\u0629 \u0625\u0644\u0649 \u0627\u0644\u0633\u0645\u0627\u0621 \u0627\u0644\u0633\u0627\u0628\u0639\u0629','id':'36831033','price':'2.80','category':'Islamic-books \/ Books \/ Research-studies','brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null,'list':'Product List','position':16}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                30%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36831033/s/5000966/category/6220/" title="لأنك الله : رحلة إلى السماء السابعة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000966_170x170.jpg" width="135" height="185" alt="لأنك الله : رحلة إلى السماء السابعة">

                                    <h4>لأنك الله : رحلة إلى السماء السابعة</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$4.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$2.80</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $4.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$2.80</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0623\u0646\u0643 \u0627\u0644\u0644\u0647 : \u0631\u062d\u0644\u0629 \u0625\u0644\u0649 \u0627\u0644\u0633\u0645\u0627\u0621 \u0627\u0644\u0633\u0627\u0628\u0639\u0629','id':'36831033','price':'2.80','category':'Islamic-books \/ Books \/ Research-studies','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36831033/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0631\u064e\u0648\u0627\u0621\u064f \u0645\u064e\u0643\u0651\u064e\u0629','id':'37173292','price':'10.00','category':'Literature-fiction \/ Books \/ Biography','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':17}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37173292/s/3109386/category/6220/" title="رَواءُ مَكَّة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3109386_170x170.jpg" width="135" height="185" alt="رَواءُ مَكَّة">

                                    <h4>رَواءُ مَكَّة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0631\u064e\u0648\u0627\u0621\u064f \u0645\u064e\u0643\u0651\u064e\u0629','id':'37173292','price':'10.00','category':'Literature-fiction \/ Books \/ Biography','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37173292/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634 : \u0627\u0644\u0643\u062a\u0627\u0628 \u0627\u0644\u0623\u0648\u0644 \u0645\u0646 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631','id':'36611542','price':'30.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':18}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3092746.html" title="لعبة العروش : الكتاب الأول من أغنية الجليد والنار" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3092746_170x170.jpg" width="135" height="185" alt="لعبة العروش : الكتاب الأول من أغنية الجليد والنار">

                                    <h4>لعبة العروش : الكتاب الأول من أغنية الجليد والنار</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$30.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $30.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634 : \u0627\u0644\u0643\u062a\u0627\u0628 \u0627\u0644\u0623\u0648\u0644 \u0645\u0646 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631','id':'36611542','price':'30.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36611542/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062e\u0627\u062a\u0645 \u0628\u0635\u0628\u0639\u064a','id':'36861849','price':'21.51','category':'Family-kids \/ Books \/ Woman-affairs','brand':'Dr. Sha3oola','variant':null,'list':'Product List','position':19}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36861849/s/4400013/category/6220/" title="خاتم بصبعي" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4400013_170x170.jpg" width="135" height="185" alt="خاتم بصبعي">

                                    <h4>خاتم بصبعي</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$21.51</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $21.51                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062e\u0627\u062a\u0645 \u0628\u0635\u0628\u0639\u064a','id':'36861849','price':'21.51','category':'Family-kids \/ Books \/ Woman-affairs','quantity':1,'brand':'Dr. Sha3oola','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36861849/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0646\u0627\u0642\u0629 \u0635\u0627\u0644\u062d\u0629','id':'37288486','price':'9.50','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null,'list':'Product List','position':20}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37288486/s/3119933/category/6220/" title="ناقة صالحة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3119933_170x170.jpg" width="135" height="185" alt="ناقة صالحة">

                                    <h4>ناقة صالحة</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $10.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$9.50</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0646\u0627\u0642\u0629 \u0635\u0627\u0644\u062d\u0629','id':'37288486','price':'9.50','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37288486/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                               <!-- End of ITEM -->

        <div class="txt-center">

                                <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    3                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

                </div>

    </div></section>





                            </div> '''
html3 = '''<div class="col-sm-9 col-md-9 col-xs-12 book-listing">

                <section class="head">

                    <div class="col-sm-8 col-md-8 col-xs-8">

                        <h1> الكتب الأكثر مبيعاً 2019</h1>

                    </div>

                </section>

                                                                            <section class="list">

        <div class="hidden-xs">

            <div class="txt-center">

                    <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    3                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

        </div>

        </div>

                        <div class="row">

                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0648\u0644\u0627\u062f \u062d\u0627\u0631\u062a\u0646\u0627','id':'2972','price':'19.60','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null,'list':'Product List','position':1}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/1006424.html" title="أولاد حارتنا" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1006424_170x170.jpg" width="135" height="185" alt="أولاد حارتنا">

                                    <h4>أولاد حارتنا</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$19.60</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $19.60                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0648\u0644\u0627\u062f \u062d\u0627\u0631\u062a\u0646\u0627','id':'2972','price':'19.60','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0634\u0631\u0648\u0642 \u2013 \u0645\u0635\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/2972/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u063a\u0646\u0649 \u0631\u062c\u0644 \u0641\u064a \u0628\u0627\u0628\u0644','id':'36624570','price':'6.09','category':'Politics \/ Self-help \/ Books','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':2}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36624570/s/5000462/category/6220/" title="أغنى رجل في بابل" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000462_170x170.jpg" width="135" height="185" alt="أغنى رجل في بابل">

                                    <h4>أغنى رجل في بابل</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.09</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $6.09                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u063a\u0646\u0649 \u0631\u062c\u0644 \u0641\u064a \u0628\u0627\u0628\u0644','id':'36624570','price':'6.09','category':'Politics \/ Self-help \/ Books','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36624570/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u0646\u062c\u062d \u0642\u0628\u0644 \u0623\u0646 \u062a\u0628\u062f\u0623','id':'36869247','price':'10.00','category':'Books \/ Best-seller-books-2019','brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null,'list':'Product List','position':3}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36869247/s/1029818/category/6220/" title="كيف تنجح قبل أن تبدأ" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1029818_170x170.jpg" width="135" height="185" alt="كيف تنجح قبل أن تبدأ">

                                    <h4>كيف تنجح قبل أن تبدأ</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u0646\u062c\u062d \u0642\u0628\u0644 \u0623\u0646 \u062a\u0628\u062f\u0623','id':'36869247','price':'10.00','category':'Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0646\u0634\u0631 \u062e\u0627\u0635','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36869247/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0639\u064a\u0634 \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u062d\u0627\u0636\u0631','id':'37297153','price':'6.40','category':'Books \/ Philosophy \/ Best-seller-books-2019','brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':4}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                20%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37297153/s/3120483/category/6220/" title="الذكاء العيش في الوقت الحاضر" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3120483_170x170.jpg" width="135" height="185" alt="الذكاء العيش في الوقت الحاضر">

                                    <h4>الذكاء العيش في الوقت الحاضر</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$8.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$6.40</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $8.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$6.40</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0639\u064a\u0634 \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u062d\u0627\u0636\u0631','id':'37297153','price':'6.40','category':'Books \/ Philosophy \/ Best-seller-books-2019','quantity':1,'brand':'\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0637\u0628\u0648\u0639\u0627\u062a \u0644\u0644\u062a\u0648\u0632\u064a\u0639 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37297153/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0631\u0648\u0627\u064a\u0629 1984','id':'268292','price':'10.00','category':'Literature-fiction \/ Novels \/ Top1oocat','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null,'list':'Product List','position':5}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3059915.html" title="رواية 1984" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3059915_170x170.jpg" width="135" height="185" alt="رواية 1984">

                                    <h4>رواية 1984</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0631\u0648\u0627\u064a\u0629 1984','id':'268292','price':'10.00','category':'Literature-fiction \/ Novels \/ Top1oocat','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/268292/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0648\u062a \u0635\u063a\u064a\u0631','id':'36761834','price':'18.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':6}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36761834/s/3094522/category/6220/" title="موت صغير" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3094522_170x170.jpg" width="135" height="185" alt="موت صغير">

                                    <h4>موت صغير</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$18.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $18.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u0648\u062a \u0635\u063a\u064a\u0631','id':'36761834','price':'18.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0633\u0627\u0642\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36761834/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u062c\u0631\u062a\u064a \u0634\u062c\u0631\u0629 \u0627\u0644\u0628\u0631\u062a\u0642\u0627\u0644 \u0627\u0644\u0631\u0627\u0626\u0639\u0629','id':'36922032','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0633\u0643\u064a\u0644\u064a\u0627\u0646\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':7}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36922032/s/1030292/category/6220/" title="شجرتي شجرة البرتقال الرائعة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1030292_170x170.jpg" width="135" height="185" alt="شجرتي شجرة البرتقال الرائعة">

                                    <h4>شجرتي شجرة البرتقال الرائعة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u062c\u0631\u062a\u064a \u0634\u062c\u0631\u0629 \u0627\u0644\u0628\u0631\u062a\u0642\u0627\u0644 \u0627\u0644\u0631\u0627\u0626\u0639\u0629','id':'36922032','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0633\u0643\u064a\u0644\u064a\u0627\u0646\u064a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36922032/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062d\u064a\u0627\u0629 \u0641\u064a \u0627\u0644\u0625\u062f\u0627\u0631\u0629','id':'362692','price':'10.00','category':'Books \/ Biography-memoir \/ Top1oocat','brand':'\u0627\u0644\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u062f\u0631\u0627\u0633\u0627\u062a \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':8}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/362692/s/3019715/category/6220/" title="حياة في الإدارة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3019715_170x170.jpg" width="135" height="185" alt="حياة في الإدارة">

                                    <h4>حياة في الإدارة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062d\u064a\u0627\u0629 \u0641\u064a \u0627\u0644\u0625\u062f\u0627\u0631\u0629','id':'362692','price':'10.00','category':'Books \/ Biography-memoir \/ Top1oocat','quantity':1,'brand':'\u0627\u0644\u0645\u0624\u0633\u0633\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u062f\u0631\u0627\u0633\u0627\u062a \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/362692/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0641\u0646 \u0623\u0646 \u062a\u0643\u0648\u0646 \u062f\u0627\u0626\u0645\u0627\u064b \u0639\u0644\u0649 \u0635\u0648\u0627\u0628','id':'36773795','price':'5.00','category':'Books \/ Philosophy \/ Top-selling-books-for-2016','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0636\u0641\u0627\u0641','variant':null,'list':'Product List','position':9}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36773795/s/3095373/category/6220/" title="فن أن تكون دائماً على صواب" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3095373_170x170.jpg" width="135" height="185" alt="فن أن تكون دائماً على صواب">

                                    <h4>فن أن تكون دائماً على صواب</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$5.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $5.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0641\u0646 \u0623\u0646 \u062a\u0643\u0648\u0646 \u062f\u0627\u0626\u0645\u0627\u064b \u0639\u0644\u0649 \u0635\u0648\u0627\u0628','id':'36773795','price':'5.00','category':'Books \/ Philosophy \/ Top-selling-books-for-2016','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0636\u0641\u0627\u0641','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36773795/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0643\u0633\u062a\u0627\u0633\u064a 65 \u064a\u0648\u0645','id':'37013627','price':'29.00','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':10}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37013627/s/5003407/category/6220/" title="اكستاسي 65 يوم" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5003407_170x170.jpg" width="135" height="185" alt="اكستاسي 65 يوم">

                                    <h4>اكستاسي 65 يوم</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$29.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $29.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0643\u0633\u062a\u0627\u0633\u064a 65 \u064a\u0648\u0645','id':'37013627','price':'29.00','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0631\u0643\u0632 \u0627\u0644\u0623\u062f\u0628 \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37013627/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u0642\u0629 \u0627\u0644\u062d\u0631\u064a\u0629','id':'379966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0631\u064a\u0627\u0636 \u0627\u0644\u0631\u064a\u0633 \u0644\u0644\u0643\u062a\u0628 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':11}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/379966/s/3004707/category/6220/" title="شقة الحرية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3004707_170x170.jpg" width="135" height="185" alt="شقة الحرية">

                                    <h4>شقة الحرية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u0642\u0629 \u0627\u0644\u062d\u0631\u064a\u0629','id':'379966','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0631\u064a\u0627\u0636 \u0627\u0644\u0631\u064a\u0633 \u0644\u0644\u0643\u062a\u0628 \u0648\u0627\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/379966/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u062c\u064a\u0646 \u0627\u0644\u0633\u0645\u0627\u0621','id':'37068467','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null,'list':'Product List','position':12}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37068467/s/3107609/category/6220/" title="سجين السماء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107609_170x170.jpg" width="135" height="185" alt="سجين السماء">

                                    <h4>سجين السماء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u062c\u064a\u0646 \u0627\u0644\u0633\u0645\u0627\u0621','id':'37068467','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37068467/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0648\u0639\u0627\u0638 \u0627\u0644\u0633\u0644\u0627\u0637\u064a\u0646','id':'428642','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Sociology','brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null,'list':'Product List','position':13}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3073341.html" title="وعاظ السلاطين" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3073341_170x170.jpg" width="135" height="185" alt="وعاظ السلاطين">

                                    <h4>وعاظ السلاطين</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$13.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $13.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0648\u0639\u0627\u0638 \u0627\u0644\u0633\u0644\u0627\u0637\u064a\u0646','id':'428642','price':'13.00','category':'Books \/ Social-sciences-humanities \/ Sociology','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/428642/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0623\u0646\u0627 \u064a\u0648\u0633\u0641','id':'37135143','price':'7.05','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0645\u0635\u0631','variant':null,'list':'Product List','position':14}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37135143/s/9008662/category/6220/" title="أنا يوسف" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9008662_170x170.jpg" width="135" height="185" alt="أنا يوسف">

                                    <h4>أنا يوسف</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.05</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.05                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0623\u0646\u0627 \u064a\u0648\u0633\u0641','id':'37135143','price':'7.05','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0645\u0635\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37135143/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0641\u0649 \u0642\u0644\u0628\u0649 \u0623\u0646\u062b\u0649 \u0639\u0628\u0631\u064a\u0629','id':'34756961','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0643\u064a\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':15}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/1017636.html" title="فى قلبى أنثى عبرية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1017636_170x170.jpg" width="135" height="185" alt="فى قلبى أنثى عبرية">

                                    <h4>فى قلبى أنثى عبرية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0641\u0649 \u0642\u0644\u0628\u0649 \u0623\u0646\u062b\u0649 \u0639\u0628\u0631\u064a\u0629','id':'34756961','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0643\u064a\u0627\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/34756961/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0623\u0646\u0643 \u0627\u0644\u0644\u0647 : \u0631\u062d\u0644\u0629 \u0625\u0644\u0649 \u0627\u0644\u0633\u0645\u0627\u0621 \u0627\u0644\u0633\u0627\u0628\u0639\u0629','id':'36831033','price':'2.80','category':'Islamic-books \/ Books \/ Research-studies','brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null,'list':'Product List','position':16}]}}});

">

                    <div class="book-wrapper">

                                                    <div class="discounted">

                                30%                                                                 خصم                            </div>

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36831033/s/5000966/category/6220/" title="لأنك الله : رحلة إلى السماء السابعة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000966_170x170.jpg" width="135" height="185" alt="لأنك الله : رحلة إلى السماء السابعة">

                                    <h4>لأنك الله : رحلة إلى السماء السابعة</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$4.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$2.80</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $4.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$2.80</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0623\u0646\u0643 \u0627\u0644\u0644\u0647 : \u0631\u062d\u0644\u0629 \u0625\u0644\u0649 \u0627\u0644\u0633\u0645\u0627\u0621 \u0627\u0644\u0633\u0627\u0628\u0639\u0629','id':'36831033','price':'2.80','category':'Islamic-books \/ Books \/ Research-studies','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062d\u0636\u0627\u0631\u0629 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 _ \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36831033/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0631\u064e\u0648\u0627\u0621\u064f \u0645\u064e\u0643\u0651\u064e\u0629','id':'37173292','price':'10.00','category':'Literature-fiction \/ Books \/ Biography','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':17}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37173292/s/3109386/category/6220/" title="رَواءُ مَكَّة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3109386_170x170.jpg" width="135" height="185" alt="رَواءُ مَكَّة">

                                    <h4>رَواءُ مَكَّة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0631\u064e\u0648\u0627\u0621\u064f \u0645\u064e\u0643\u0651\u064e\u0629','id':'37173292','price':'10.00','category':'Literature-fiction \/ Books \/ Biography','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37173292/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634 : \u0627\u0644\u0643\u062a\u0627\u0628 \u0627\u0644\u0623\u0648\u0644 \u0645\u0646 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631','id':'36611542','price':'30.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':18}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3092746.html" title="لعبة العروش : الكتاب الأول من أغنية الجليد والنار" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3092746_170x170.jpg" width="135" height="185" alt="لعبة العروش : الكتاب الأول من أغنية الجليد والنار">

                                    <h4>لعبة العروش : الكتاب الأول من أغنية الجليد والنار</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$30.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $30.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u0639\u0628\u0629 \u0627\u0644\u0639\u0631\u0648\u0634 : \u0627\u0644\u0643\u062a\u0627\u0628 \u0627\u0644\u0623\u0648\u0644 \u0645\u0646 \u0623\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0644\u064a\u062f \u0648\u0627\u0644\u0646\u0627\u0631','id':'36611542','price':'30.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36611542/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062e\u0627\u062a\u0645 \u0628\u0635\u0628\u0639\u064a','id':'36861849','price':'21.51','category':'Family-kids \/ Books \/ Woman-affairs','brand':'Dr. Sha3oola','variant':null,'list':'Product List','position':19}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36861849/s/4400013/category/6220/" title="خاتم بصبعي" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4400013_170x170.jpg" width="135" height="185" alt="خاتم بصبعي">

                                    <h4>خاتم بصبعي</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$21.51</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $21.51                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062e\u0627\u062a\u0645 \u0628\u0635\u0628\u0639\u064a','id':'36861849','price':'21.51','category':'Family-kids \/ Books \/ Woman-affairs','quantity':1,'brand':'Dr. Sha3oola','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/36861849/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0646\u0627\u0642\u0629 \u0635\u0627\u0644\u062d\u0629','id':'37288486','price':'9.50','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null,'list':'Product List','position':20}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37288486/s/3119933/category/6220/" title="ناقة صالحة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3119933_170x170.jpg" width="135" height="185" alt="ناقة صالحة">

                                    <h4>ناقة صالحة</h4>

                                    

                                    <div class="old-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.50</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                            $10.00                                                                                    </li>

                                        <li class="new-price">

                                            &nbsp;<span class="num">$9.50</span>

                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0646\u0627\u0642\u0629 \u0635\u0627\u0644\u062d\u0629','id':'37288486','price':'9.50','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD0z/product/37288486/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                               <!-- End of ITEM -->

        <div class="txt-center">

                                <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    3                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=5">

                            5                        </a>

                    </span>

                    </li>

                                                                                <li>

                    <a class="next i-next" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة التالية">

                                                    <span>»</span>

                                            </a>

                </li>

                    </ul>

                </div>

    </div></section>





                            </div> '''
html5 = '''<div class="col-sm-9 col-md-9 col-xs-12 book-listing">

                <section class="head">

                    <div class="col-sm-8 col-md-8 col-xs-8">

                        <h1> الكتب الأكثر مبيعاً 2019</h1>

                    </div>

                </section>

                                                                            <section class="list">

        <div class="hidden-xs">

            <div class="txt-center">

                    <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    5                    </span>

                    </li>

                                                                        </ul>

        </div>

        </div>

                        <div class="row">

                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062d\u062f\u064a\u062b \u0627\u0644\u0635\u0628\u0627\u062d','id':'36626308','price':'12.00','category':'Literature-fiction \/ Books \/ Literary-texts','brand':'\u062f\u0627\u0631 \u0643\u0644\u0645\u0627\u062a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0627\u0644\u0643\u0648\u064a\u062a','variant':null,'list':'Product List','position':1}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36626308/s/4100319/category/6220/" title="حديث الصباح" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4100319_170x170.jpg" width="135" height="185" alt="حديث الصباح">

                                    <h4>حديث الصباح</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062d\u062f\u064a\u062b \u0627\u0644\u0635\u0628\u0627\u062d','id':'36626308','price':'12.00','category':'Literature-fiction \/ Books \/ Literary-texts','quantity':1,'brand':'\u062f\u0627\u0631 \u0643\u0644\u0645\u0627\u062a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0627\u0644\u0643\u0648\u064a\u062a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36626308/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0645\u0632\u062f\u064e\u0648\u064e\u062c','id':'36944389','price':'8.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':2}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36944389/s/3105342/category/6220/" title="المزدَوَج" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3105342_170x170.jpg" width="135" height="185" alt="المزدَوَج">

                                    <h4>المزدَوَج</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$8.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $8.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0645\u0632\u062f\u064e\u0648\u064e\u062c','id':'36944389','price':'8.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36944389/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u062a\u062e\u0644\u0641 \u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639\u064a \u0645\u062f\u062e\u0644 \u0625\u0644\u0649 \u0633\u064a\u0643\u0648\u0644\u0648\u062c\u064a\u0629 \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0627\u0644\u0645\u0642\u0647\u0648\u0631','id':'290100','price':'16.00','category':'Social-sciences-humanities \/ Sociology \/ Top-selling-books-for-2016','brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null,'list':'Product List','position':3}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/290100/s/3059871/category/6220/" title="التخلف الاجتماعي مدخل إلى سيكولوجية الإنسان المقهور" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3059871_170x170.jpg" width="135" height="185" alt="التخلف الاجتماعي مدخل إلى سيكولوجية الإنسان المقهور">

                                    <h4>التخلف الاجتماعي مدخل إلى سيكولوجية الإنسان المقهور</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$16.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $16.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u062a\u062e\u0644\u0641 \u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639\u064a \u0645\u062f\u062e\u0644 \u0625\u0644\u0649 \u0633\u064a\u0643\u0648\u0644\u0648\u062c\u064a\u0629 \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0627\u0644\u0645\u0642\u0647\u0648\u0631','id':'290100','price':'16.00','category':'Social-sciences-humanities \/ Sociology \/ Top-selling-books-for-2016','quantity':1,'brand':'\u0627\u0644\u0645\u0631\u0643\u0632 \u0627\u0644\u062b\u0642\u0627\u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/290100/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0633\u062d\u0631 \u0627\u0644\u062a\u0631\u062a\u064a\u0628 : \u0627\u0644\u0641\u0646 \u0627\u0644\u064a\u0627\u0628\u0627\u0646\u064a \u0641\u064a \u0627\u0644\u062a\u0646\u0638\u064a\u0645 \u0648\u0625\u0632\u0627\u0644\u0629 \u0627\u0644\u0641\u0648\u0636\u0649','id':'36769641','price':'12.00','category':'Literature-fiction \/ Self-help \/ Books','brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null,'list':'Product List','position':4}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36769641/s/3094958/category/6220/" title="سحر الترتيب : الفن الياباني في التنظيم وإزالة الفوضى" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3094958_170x170.jpg" width="135" height="185" alt="سحر الترتيب : الفن الياباني في التنظيم وإزالة الفوضى">

                                    <h4>سحر الترتيب : الفن الياباني في التنظيم وإزالة الفوضى</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0633\u062d\u0631 \u0627\u0644\u062a\u0631\u062a\u064a\u0628 : \u0627\u0644\u0641\u0646 \u0627\u0644\u064a\u0627\u0628\u0627\u0646\u064a \u0641\u064a \u0627\u0644\u062a\u0646\u0638\u064a\u0645 \u0648\u0625\u0632\u0627\u0644\u0629 \u0627\u0644\u0641\u0648\u0636\u0649','id':'36769641','price':'12.00','category':'Literature-fiction \/ Self-help \/ Books','quantity':1,'brand':'\u0627\u0644\u062f\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0644\u0644\u0639\u0644\u0648\u0645 \u0646\u0627\u0634\u0631\u0648\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36769641/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u062a\u0642\u0646 \u0644\u0639\u0628\u0629 \u0627\u0644\u062d\u064a\u0627\u0629 : \u062d\u0644\u0648\u0644 \u0644 369 \u0645\u0634\u0643\u0644\u0629','id':'37016319','price':'19.07','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0645\u0644\u0647\u0645\u0648\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':5}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37016319/s/4202453/category/6220/" title="كيف تتقن لعبة الحياة : حلول ل 369 مشكلة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4202453_170x170.jpg" width="135" height="185" alt="كيف تتقن لعبة الحياة : حلول ل 369 مشكلة">

                                    <h4>كيف تتقن لعبة الحياة : حلول ل 369 مشكلة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$19.07</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $19.07                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0643\u064a\u0641 \u062a\u062a\u0642\u0646 \u0644\u0639\u0628\u0629 \u0627\u0644\u062d\u064a\u0627\u0629 : \u062d\u0644\u0648\u0644 \u0644 369 \u0645\u0634\u0643\u0644\u0629','id':'37016319','price':'19.07','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0645\u0644\u0647\u0645\u0648\u0646 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/37016319/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062e\u0648\u0627\u0631\u0642 \u0627\u0644\u0644\u0627\u0634\u0639\u0648\u0631 : \u0623\u0648 \u0623\u0633\u0631\u0627\u0631 \u0627\u0644\u0634\u062e\u0635\u064a\u0629 \u0627\u0644\u0646\u0627\u062c\u062d\u0629','id':'364176','price':'11.00','category':'Literature-fiction \/ Books \/ Social-sciences-humanities','brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null,'list':'Product List','position':6}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3016259.html" title="خوارق اللاشعور : أو أسرار الشخصية الناجحة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3016259_170x170.jpg" width="135" height="185" alt="خوارق اللاشعور : أو أسرار الشخصية الناجحة">

                                    <h4>خوارق اللاشعور : أو أسرار الشخصية الناجحة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $11.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u062e\u0648\u0627\u0631\u0642 \u0627\u0644\u0644\u0627\u0634\u0639\u0648\u0631 : \u0623\u0648 \u0623\u0633\u0631\u0627\u0631 \u0627\u0644\u0634\u062e\u0635\u064a\u0629 \u0627\u0644\u0646\u0627\u062c\u062d\u0629','id':'364176','price':'11.00','category':'Literature-fiction \/ Books \/ Social-sciences-humanities','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0648\u0631\u0627\u0642 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u2013 \u0644\u0628\u0646\u0627\u0646','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/364176/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0638\u0644 \u0627\u0644\u0631\u064a\u062d','id':'36773807','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null,'list':'Product List','position':7}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36773807/s/3095377/category/6220/" title="ظل الريح" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3095377_170x170.jpg" width="135" height="185" alt="ظل الريح">

                                    <h4>ظل الريح</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0638\u0644 \u0627\u0644\u0631\u064a\u062d','id':'36773807','price':'12.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0645\u0646\u0634\u0648\u0631\u0627\u062a \u0627\u0644\u062c\u0645\u0644','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36773807/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0625\u0633\u0623\u0644 \u062a\u0639\u0637 : \u062a\u0639\u0644\u0645 \u0623\u0646 \u062a\u0638\u0647\u0631 \u0631\u063a\u0628\u0627\u062a\u0643','id':'36792605','price':'14.00','category':'Literature-fiction \/ Books \/ Social-sciences-humanities','brand':'\u062f\u0627\u0631 \u0627\u0644\u062e\u064a\u0627\u0644 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':8}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36792605/s/3097227/category/6220/" title="إسأل تعط : تعلم أن تظهر رغباتك" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3097227_170x170.jpg" width="135" height="185" alt="إسأل تعط : تعلم أن تظهر رغباتك">

                                    <h4>إسأل تعط : تعلم أن تظهر رغباتك</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$14.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $14.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0625\u0633\u0623\u0644 \u062a\u0639\u0637 : \u062a\u0639\u0644\u0645 \u0623\u0646 \u062a\u0638\u0647\u0631 \u0631\u063a\u0628\u0627\u062a\u0643','id':'36792605','price':'14.00','category':'Literature-fiction \/ Books \/ Social-sciences-humanities','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062e\u064a\u0627\u0644 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36792605/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0648\u0644\u064a\u0645\u0629 \u0644\u0644\u063a\u0631\u0628\u0627\u0646 : 1 \u2013 2','id':'37092467','price':'30.00','category':'Science-fiction-and-fantasy \/ Books \/ Best-seller-books-2019','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':9}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37092467/s/3107807/category/6220/" title="وليمة للغربان : 1 – 2" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3107807_170x170.jpg" width="135" height="185" alt="وليمة للغربان : 1 – 2">

                                    <h4>وليمة للغربان : 1 – 2</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$30.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $30.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0648\u0644\u064a\u0645\u0629 \u0644\u0644\u063a\u0631\u0628\u0627\u0646 : 1 \u2013 2','id':'37092467','price':'30.00','category':'Science-fiction-and-fantasy \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/37092467/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0639\u0627\u0644\u0645 \u0635\u0648\u0641\u064a : \u0631\u0648\u0627\u064a\u0629 \u062d\u0648\u0644 \u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0641\u0644\u0633\u0641\u0629','id':'384276','price':'23.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0646\u0649','variant':null,'list':'Product List','position':10}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/3016379.html" title="عالم صوفي : رواية حول تاريخ الفلسفة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3016379_170x170.jpg" width="135" height="185" alt="عالم صوفي : رواية حول تاريخ الفلسفة">

                                    <h4>عالم صوفي : رواية حول تاريخ الفلسفة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$23.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $23.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0639\u0627\u0644\u0645 \u0635\u0648\u0641\u064a : \u0631\u0648\u0627\u064a\u0629 \u062d\u0648\u0644 \u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0641\u0644\u0633\u0641\u0629','id':'384276','price':'23.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0645\u0646\u0649','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/384276/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0634\u064a\u0641\u0631\u0629 \u0628\u0644\u0627\u0644','id':'36825767','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':11}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36825767/s/9005111/category/6220/" title="شيفرة بلال" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9005111_170x170.jpg" width="135" height="185" alt="شيفرة بلال">

                                    <h4>شيفرة بلال</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$10.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $10.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0634\u064a\u0641\u0631\u0629 \u0628\u0644\u0627\u0644','id':'36825767','price':'10.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36825767/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0644\u064a\u0637\u0645\u0626\u0646 \u0639\u0642\u0644\u064a','id':'37136360','price':'20.00','category':'Self-help \/ Books \/ Best-seller-books-2019','brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':12}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37136360/s/9008703/category/6220/" title="ليطمئن عقلي" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/9008703_170x170.jpg" width="135" height="185" alt="ليطمئن عقلي">

                                    <h4>ليطمئن عقلي</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$20.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $20.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0644\u064a\u0637\u0645\u0626\u0646 \u0639\u0642\u0644\u064a','id':'37136360','price':'20.00','category':'Self-help \/ Books \/ Best-seller-books-2019','quantity':1,'brand':'\u0639\u0635\u064a\u0631 \u0627\u0644\u0643\u062a\u0628 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/37136360/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0639\u0627\u062f\u0627\u062a \u0627\u0644\u0633\u0628\u0639 \u0644\u0644\u0646\u0627\u0633 \u0627\u0644\u0623\u0643\u062b\u0631 \u0641\u0639\u0627\u0644\u064a\u0629','id':'430120','price':'19.46','category':'Management-and-development \/ Self-help \/ Books','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':13}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/430120/s/1014834/category/6220/" title="العادات السبع للناس الأكثر فعالية" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/1014834_170x170.jpg" width="135" height="185" alt="العادات السبع للناس الأكثر فعالية">

                                    <h4>العادات السبع للناس الأكثر فعالية</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$19.46</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $19.46                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0639\u0627\u062f\u0627\u062a \u0627\u0644\u0633\u0628\u0639 \u0644\u0644\u0646\u0627\u0633 \u0627\u0644\u0623\u0643\u062b\u0631 \u0641\u0639\u0627\u0644\u064a\u0629','id':'430120','price':'19.46','category':'Management-and-development \/ Self-help \/ Books','quantity':1,'brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/430120/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0626\u0629 \u0639\u0627\u0645\u064d \u0645\u0646 \u0627\u0644\u0639\u0632\u0644\u0629','id':'36854713','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':14}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36854713/s/3101432/category/6220/" title="مئة عامٍ من العزلة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3101432_170x170.jpg" width="135" height="185" alt="مئة عامٍ من العزلة">

                                    <h4>مئة عامٍ من العزلة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$15.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $15.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0645\u0626\u0629 \u0639\u0627\u0645\u064d \u0645\u0646 \u0627\u0644\u0639\u0632\u0644\u0629','id':'36854713','price':'15.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u062a\u0646\u0648\u064a\u0631 \u0644\u0644\u0637\u0628\u0627\u0639\u0629 \u0648\u0627\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/36854713/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0628\u0646\u062a \u0627\u0644\u062a\u064a \u0644\u0627 \u062a\u062d\u0628 \u0627\u0633\u0645\u0647\u0627','id':'37174970','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null,'list':'Product List','position':15}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/37174970/s/3109504/category/6220/" title="البنت التي لا تحب اسمها" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3109504_170x170.jpg" width="135" height="185" alt="البنت التي لا تحب اسمها">

                                    <h4>البنت التي لا تحب اسمها</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0628\u0646\u062a \u0627\u0644\u062a\u064a \u0644\u0627 \u062a\u062d\u0628 \u0627\u0633\u0645\u0647\u0627','id':'37174970','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','quantity':1,'brand':'\u062f\u0627\u0631 \u0627\u0644\u0622\u062f\u0627\u0628','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/37174970/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0627\u0644\u0631\u0642\u0635 \u0645\u0639 \u0627\u0644\u062d\u064a\u0627\u0629','id':'35097159','price':'9.00','category':'Literature-fiction \/ Self-help \/ Books','brand':'\u062f\u0627\u0631 \u0645\u062f\u0627\u0631\u0643 \u0644\u0644\u0646\u0634\u0631','variant':null,'list':'Product List','position':16}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/35097159/s/3088998/category/6220/" title="الرقص مع الحياة" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3088998_170x170.jpg" width="135" height="185" alt="الرقص مع الحياة">

                                    <h4>الرقص مع الحياة</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$9.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $9.00                                                                                    </li>

                                    </ul>

                                                                            <p>

                                            <button type="button" title="أضف إلى عربة التسوق" class="button btn-cart btn btn-default add-to-cart" onclick="

dataLayer.push({'event':'addToCart','ecommerce':{'currencyCode':'USD','add':{'products':[{'name':'\u0627\u0644\u0631\u0642\u0635 \u0645\u0639 \u0627\u0644\u062d\u064a\u0627\u0629','id':'35097159','price':'9.00','category':'Literature-fiction \/ Self-help \/ Books','quantity':1,'brand':'\u062f\u0627\u0631 \u0645\u062f\u0627\u0631\u0643 \u0644\u0644\u0646\u0634\u0631','variant':null}]}}});



; setLocation('https://jamalon.com/ar/checkout/cart/add/uenc/aHR0cHM6Ly9qYW1hbG9uLmNvbS9hci9iZXN0LXNlbGxlci1ib29rcy0yMDE5P19fX2Zyb21fc3RvcmU9ZW4mcD01/product/35097159/')">

                                                <span><span>أضف إلى عربة التسوق</span></span>

                                            </button>

                                        </p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u062c\u0648 : \u0639\u0630\u0627\u0628\u0627\u062a 10 \u0645\u0627\u0631\u0633 \u0641\u064a \u0633\u062c\u0646 \u062c\u0648','id':'36799295','price':'7.00','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0631\u0622\u0629 \u0627\u0644\u0628\u062d\u0631\u064a\u0646','variant':null,'list':'Product List','position':17}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36799295/s/3097546/category/6220/" title="جو : عذابات 10 مارس في سجن جو" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3097546_170x170.jpg" width="135" height="185" alt="جو : عذابات 10 مارس في سجن جو">

                                    <h4>جو : عذابات 10 مارس في سجن جو</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.00                                                                                    </li>

                                    </ul>

                                                                            <p class="availability out-of-stock"><span>Out of Stock </span></p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0648\u0627\u062e\u062a\u0641\u0649 \u0643\u0644 \u0634\u0649\u0621','id':'36790271','price':'7.20','category':'Literature-fiction \/ Books \/ Novels','brand':'\u0645\u0643\u062a\u0628\u0629 \u062c\u0631\u064a\u0631','variant':null,'list':'Product List','position':18}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36790271/s/5000742/category/6220/" title="واختفى كل شىء" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/5000742_170x170.jpg" width="135" height="185" alt="واختفى كل شىء">

                                    <h4>واختفى كل شىء</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$7.20</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $7.20                                                                                    </li>

                                    </ul>

                                                                            <p class="availability out-of-stock"><span>Out of Stock </span></p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                </div><div class="row">                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0627\u062f\u0648\u0646\u0627 \u0635\u0627\u062d\u0628\u0629 \u0645\u0639\u0637\u0641 \u0627\u0644\u0641\u0631\u0648','id':'36819055','price':'11.00','category':'Literature-fiction \/ Books \/ Best-seller-books-2019','brand':'\u062f\u0627\u0631 \u0623\u062b\u0631 \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639','variant':null,'list':'Product List','position':19}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36819055/s/3098276/category/6220/" title="مادونا صاحبة معطف الفرو" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/3098276_170x170.jpg" width="135" height="185" alt="مادونا صاحبة معطف الفرو">

                                    <h4>مادونا صاحبة معطف الفرو</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$11.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $11.00                                                                                    </li>

                                    </ul>

                                                                            <p class="availability out-of-stock"><span>Out of Stock </span></p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                                                             <div class="col-sm-4 col-md-4 col-xs-12" onclick="

dataLayer.push({'event':'productClick','ecommerce':{'click':{'products':[{'name':'\u0645\u0644\u0647\u0645\u0648\u0646','id':'36770981','price':'12.00','category':'Literature-fiction \/ Books \/ Top-selling-books-for-2016','brand':'\u062f\u0627\u0631 \u0643\u0644\u0645\u0627\u062a \u0644\u0644\u0646\u0634\u0631 \u0648\u0627\u0644\u062a\u0648\u0632\u064a\u0639 - \u0627\u0644\u0643\u0648\u064a\u062a','variant':null,'list':'Product List','position':20}]}}});

">

                    <div class="book-wrapper">

                                                <div class="book">

                            <a href="https://jamalon.com/ar/catalog/product/view/id/36770981/s/4100558/category/6220/" title="ملهمون" class="product-image">

                                <div class="book-info text-center">

                                    <img src="https://cdn.jamalon.com/c/p/4100558_170x170.jpg" width="135" height="185" alt="ملهمون">

                                    <h4>ملهمون</h4>

                                    

                                    <div class="old-price">

                                                                            </div>

                                    <div class="new-price">

                                                                                    <span class="num">$12.00</span>

                                                                            </div>

                                </div>

                            </a>

                            <div class="book-overlay">

                                <div class="overlay-content">

                                    <ul class="prices list-inline">

                                        <li class="old-price">

                                                                                    </li>

                                        <li class="new-price">

                                                                                            $12.00                                                                                    </li>

                                    </ul>

                                                                            <p class="availability out-of-stock"><span>Out of Stock </span></p>

                                                                    </div>

                            </div>

                        </div>

                    </div> <!-- Book wrapper -->

                </div><!-- col 12 -->

                               <!-- End of ITEM -->

        <div class="txt-center">

                                <ul class="pagination">

                            <li>

                    <a class="previous i-previous" href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4" title="الصفحة السابقة">

                                                    <span>«</span>

                                            </a>

                </li>

                                                                                    <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=1">

                            1                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=2">

                            2                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=3">

                            3                        </a>

                    </span>

                    </li>

                                                                <li>

                    <span>

                        <a href="https://jamalon.com/ar/best-seller-books-2019?___from_store=en&amp;p=4">

                            4                        </a>

                    </span>

                    </li>

                                                                <li class="active">

                    <span>

                    5                    </span>

                    </li>

                                                                        </ul>

                </div>

    </div></section>





                            </div> '''
# create browser obj

driver = webdriver.Chrome('chromedriver/chromedriver')

# request url

driver.get('https://jamalon.com/ar/best-seller-books-2019?___from_store=en&p=1')



#use for loop to looping throw the pages

for i in range(2,7): 

    

    clicker = WebDriverWait(driver, 40).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="content"]/section/div/div/div[2]/section[2]/div[8]/div[3]/ul/li[{i}]/span')))

    clicker.click()
# step 2 :parsing the all page

soup1 = BeautifulSoup(html1, 'html.parser')

soup2 = BeautifulSoup(html2, 'html.parser')

soup3 = BeautifulSoup(html3, 'html.parser')

soup4 = BeautifulSoup(html4, 'html.parser')

soup5 = BeautifulSoup(html5, 'html.parser')
#here the books info from each page

page1=soup1.find_all('div', attrs={'class':'row'})

page2=soup2.find_all('div', attrs={'class':'row'})

page3=soup3.find_all('div', attrs={'class':'row'})

page4=soup4.find_all('div', attrs={'class':'row'})

page5=soup5.find_all('div', attrs={'class':'row'})

print(page1)
# here i choose only the first item

items1 =soup1.find_all('div', attrs={'class':'book-info text-center'})

items2 =soup2.find_all('div', attrs={'class':'book-info text-center'})

items3 =soup3.find_all('div', attrs={'class':'book-info text-center'})

items4 =soup4.find_all('div', attrs={'class':'book-info text-center'})

items5 =soup5.find_all('div', attrs={'class':'book-info text-center'})

print (items1[0])
# extract only the text from book name and book price

book=items1[0].find('h4').get_text()

print(book)

price=items1[0].find(class_='num').get_text()

print(price)
# use list comprehension for all elements

books_names1=[item.find('h4').get_text()for item in items1]

books_prices1=[item.find(class_='num').get_text()for item in items1]





books_names2=[item.find('h4').get_text()for item in items2]

books_prices2=[item.find(class_='num').get_text()for item in items2]





books_names3=[item.find('h4').get_text()for item in items3]

books_prices3=[item.find(class_='num').get_text()for item in items3]





books_names4=[item.find('h4').get_text()for item in items4]

books_prices4=[item.find(class_='num').get_text()for item in items4]





books_names5=[item.find('h4').get_text()for item in items5]

books_prices5=[item.find(class_='num').get_text()for item in items5]

Best_Seller_Books1 =pd.DataFrame({'book_name':books_names1,'price_in_US':books_prices1})

Best_Seller_Books2 =pd.DataFrame({'book_name':books_names2,'price_in_US':books_prices2})

Best_Seller_Books3 =pd.DataFrame({'book_name':books_names3,'price_in_US':books_prices3})

Best_Seller_Books4 =pd.DataFrame({'book_name':books_names4,'price_in_US':books_prices4})

Best_Seller_Books5 =pd.DataFrame({'book_name':books_names5,'price_in_US':books_prices5})
frames= [Best_Seller_Books1, Best_Seller_Books2, Best_Seller_Books3,Best_Seller_Books4,Best_Seller_Books5]



Best_Seller= pd.concat(frames)
Best_Seller=Best_Seller.reset_index()

Best_Seller=Best_Seller.filter(items=['price_in_US','book_name'])

Best_Seller
Best_Seller.dtypes
Best_Seller.shape
Best_Seller['price_in_US'] = Best_Seller['price_in_US'].map(lambda x: x.strip('$'))

Best_Seller['price_in_US'] = pd.to_numeric(Best_Seller['price_in_US'])
Best_Seller['price_in_US']
Best_Seller.dtypes
#deleting duplicate rows

Best_Seller=Best_Seller.drop(Best_Seller.index[20])

Best_Seller=Best_Seller.drop(Best_Seller.index[80])

Best_Seller.head(50)
Best_Seller.describe() 
Best_Seller.sort_values('price_in_US', ascending=False).head(10)
conditions = [(Best_Seller['price_in_US'] >=3) & (Best_Seller['price_in_US'] <= 9),(Best_Seller['price_in_US'] >9) & (Best_Seller['price_in_US'] <= 16),

             (Best_Seller['price_in_US'] >16)]
choices = ['cheap','Average_price','Expensive']
Best_Seller['Category'] = np.select(condlist=conditions, choicelist=choices)
Best_Seller
import plotly

import pandas as pd

import plotly.graph_objs as go
Best_Seller=Best_Seller.drop('Unnamed: 0', axis = 1)

Best_Seller.head()
a=Best_Seller.loc[(Best_Seller['price_in_US'] >=3) & (Best_Seller['price_in_US'] <= 9)]

b=Best_Seller.loc[(Best_Seller['price_in_US'] >9) & (Best_Seller['price_in_US'] <= 16)]

c=Best_Seller.loc[(Best_Seller['price_in_US'] >16)]



Cat=['Cheap','Average_price','Expensive']



fig = go.Figure([go.Bar(x=Cat,  y=[len(a), len(b), len(c)])])



fig.update_layout(autosize=False,

    width=600,

    height=500,

                  title_text="Distribution of books in each category",

                  xaxis_title = "Categories",

                  yaxis_title = "Number of books")



fig.show()
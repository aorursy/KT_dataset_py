import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
html = """

<html lang="en">

<head>

    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable:no" />

    <meta http-equiv="X-UA-Compatible" content="IE=edge" />

    



<title>Facts and figures: Ending violence against women | UN Women &ndash; Headquarters</title>



<meta name="description" content="Facts and figures: Ending violence against women" />

<meta name="keywords" content="" />

<meta property="og:title" content="Facts and figures: Ending violence against women" />

<meta property="og:description" content="Facts and figures: Ending violence against women" />

<meta property="og:type" content="article" />

<meta property="og:image" content="https://www.unwomen.org/-/media/un%20women%20logos/un-women-logo-social-media-1024x512-en.png?vs=2759" />

<meta property="og:url" content="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures" />

<meta property="og:site_name" content="UN Women" />

<meta property="fb:app_id" content="488769704655473" />

    <meta name="twitter:site" content="@UN_Women" />

<meta name="twitter:card" content="summary_large_image" />

<meta name="twitter:image" content="https://www.unwomen.org/-/media/un%20women%20logos/un-women-logo-social-media-1024x512-en.png?vs=2759" />





    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />

    <link id="page_favicon" href="https://www.unwomen.org/images/favicons/unw.ico" rel="icon" type="image/x-icon">

    <script type="text/javascript" src="https://www.unwomen.org/Scripts/communications/modernizr.min.js?d=2020-06-25T05:51:44"></script>

    <script type="text/javascript" src="https://www.unwomen.org/Scripts/communications/jquery.min.js?d=2020-06-25T05:52:00"></script>

    

<!-- start:fonts -->



<!-- end:fonts -->

<link href="https://www.unwomen.org/Styles/communications/reset.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Scripts/communications/fancybox/jquery.fancybox.css?d=2020-06-25T05:51:50" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/tooltipster.bundle.min.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/hq.css?d=2020-06-25T05:52:02" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/dev_overrides.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/style.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/default.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/component.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/search.css?d=2020-06-25T05:51:44" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/breakingNews.css?d=2020-06-25T05:51:42" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/MixMedia/jquery.bxslider.css?d=2020-06-25T05:51:50" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/MixMedia/jquery.bxslider-custom.css?d=2020-06-25T05:51:48" rel="stylesheet">

<link href="https://www.unwomen.org/Styles/communications/MixMedia/mixed-media.css?d=2020-06-25T05:51:50" rel="stylesheet">



<link href="https://www.unwomen.org/Styles/communications/hq-media-queries.css?d=2020-06-25T05:51:44" rel="stylesheet">









<link href="/Styles/communications/print.css" rel="stylesheet" type="text/css" media="print" />

    



<!-- Global site tag (gtag.js) - Google Analytics -->

<script async src="https://www.googletagmanager.com/gtag/js?id=UA-3482969-13"></script>

<script>

  window.dataLayer = window.dataLayer || [];

  function gtag(){dataLayer.push(arguments);}

  gtag('js', new Date());



  gtag('config', 'UA-3482969-13');

</script>



        <!--

         To collect end-user usage analytics about your application,

         insert the following script into each page you want to track.

         Place this code immediately before the closing </head> tag,

         and before any other scripts. Your first data will appear

         automatically in just a few seconds.

         -->

        <script type="text/javascript">

                var appInsights = window.appInsights || function (config) {

                    function i(config) { t[config] = function () { var i = arguments; t.queue.push(function () { t[config].apply(t, i) }) } } var t = { config: config }, u = document, e = window, o = "script", s = "AuthenticatedUserContext", h = "start", c = "stop", l = "Track", a = l + "Event", v = l + "Page", y = u.createElement(o), r, f; y.src = config.url || "https://az416426.vo.msecnd.net/scripts/a/ai.0.js"; u.getElementsByTagName(o)[0].parentNode.appendChild(y); try { t.cookie = u.cookie } catch (p) { } for (t.queue = [], t.version = "1.0", r = ["Event", "Exception", "Metric", "PageView", "Trace", "Dependency"]; r.length;) i("track" + r.pop()); return i("set" + s), i("clear" + s), i(h + a), i(c + a), i(h + v), i(c + v), i("flush"), config.disableExceptionTracking || (r = "onerror", i("_" + r), f = e[r], e[r] = function (config, i, u, e, o) { var s = f && f(config, i, u, e, o); return s !== !0 && t["_" + r](config, i, u, e, o), s }), t

                }({

                    instrumentationKey: "9df0fd13-d4dd-449e-9e69-265936ca26da",

                    disableAjaxTracking: true

                });



                window.appInsights = appInsights;

                appInsights.trackPageView(null, null, { urlReferrer: document.referrer });

        </script>



    <!-- Google Tag Manager -->

        <script>

            (function (w, d, s, l, i) {

            w[l] = w[l] || []; w[l].push({

                'gtm.start':

                new Date().getTime(), event: 'gtm.js'

            }); var f = d.getElementsByTagName(s)[0],

                j = d.createElement(s), dl = l != 'dataLayer' ? '&l=' + l : ''; j.async = true; j.src =

                    'https://www.googletagmanager.com/gtm.js?id=' + i + dl; f.parentNode.insertBefore(j, f);

            })(window, document, 'script', 'dataLayer', 'GTM-TH8N5R7');</script>

    <!-- End Google Tag Manager -->

</head>

<body>

    <!-- Google Tag Manager (noscript) -->

        <noscript>

            <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TH8N5R7"

                    height="0" width="0" style="display:none;visibility:hidden"></iframe>

        </noscript>

    <!-- End Google Tag Manager (noscript) -->

    <div id="language-bar-content" style="display:none">

        





<ul class="translation">

            <li>

                <a data-language="en" href="/en/what-we-do/ending-violence-against-women/facts-and-figures">English</a>

            </li>

            <li>

                <a data-language="es" href="/es/what-we-do/ending-violence-against-women/facts-and-figures">Español</a>

            </li>

            <li>

                <a data-language="fr" href="/fr/what-we-do/ending-violence-against-women/facts-and-figures">Français</a>

            </li>

</ul>





    </div>

    <!-- Header starts -->







<div class="hq-masthead content-wrapper">

            <div id="language-bar-place-holder">&nbsp;</div>



    <div class="search" role="search">

        <div>

    <input name="headerSearchTxt" type="text" id="headerSearchTxt" class="search-input placeholder-text" placeholder="Search...">

    <input type="button" name="btnSearch" value="Search" id="btnSearch" class="png_bg search-submit">

    <span style="color: Red; display: none;"></span>

</div>





    </div>

    <div class="masthead-container">

        <h1 class="logo">

            <a href="https://www.unwomen.org/en">

                <img src="https://www.unwomen.org/-/media/un%20women%20logos/unwomen-logo-blue-transparent-background-247x70-en.gif?vs=920&amp;h=70&amp;w=247&amp;la=en&amp;hash=B0B7F936CE554C093922FECB7AB421881795B1BA" alt="UN&#32;Women&#32;&#8211;&#32;United&#32;Nations&#32;Entity&#32;for&#32;Gender&#32;Equality&#32;and&#32;the&#32;Empowerment&#32;of&#32;Women" />

            </a>

        </h1>

        <h1 class="logo" style="display: none;">

            <a href="/">

                <img src="https://www.unwomen.org/-/media/un%20women%20logos/unwomen-logo-blue-transparent-background-247x70-en.gif?vs=920&amp;h=70&amp;w=247&amp;la=en&amp;hash=B0B7F936CE554C093922FECB7AB421881795B1BA" alt="UN&#32;Women&#32;&#8211;&#32;United&#32;Nations&#32;Entity&#32;for&#32;Gender&#32;Equality&#32;and&#32;the&#32;Empowerment&#32;of&#32;Women" />

            </a>

        </h1>

        <h2 class="masthead masthead-lrg">

            

        </h2>



    </div>



    





<div class="top-nevigation">

    <ul class="top-nevigation-menu">

            <li>

                <a class="multi-line&#32;ga-event" data-action="ui-topnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/executive-board" title="Executive&#32;Board">Executive<span class="top-nevigation-menu-wrap"> Board</span></a>

                



                    <ul class="topdropToLeft">

                            <li>



                                <a title="Strategic Plan" href="https://www.unwomen.org/en/executive-board/strategic-plan" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Strategic Plan</a>



                                    <ul>

                                            <li>

                                                <a title="Impact Area" href="https://www.unwomen.org/en/executive-board/strategic-plan/impact-area" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Impact Area</a>

                                            </li>

                                            <li>

                                                <a title="Outcome Area 1" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-1" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Outcome Area 1</a>

                                            </li>

                                            <li>

                                                <a title="Outcome Area 2" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-2" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Outcome Area 2</a>

                                            </li>

                                            <li>

                                                <a title="Outcome Area 3" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-3" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Outcome Area 3</a>

                                            </li>

                                            <li>

                                                <a title="Outcome Area 4" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-4" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Outcome Area 4</a>

                                            </li>

                                            <li>

                                                <a title="Outcome Area 5" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-5" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Outcome Area 5</a>

                                            </li>

                                            <li>

                                                <a title="OEE-1" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-1" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">OEE-1</a>

                                            </li>

                                            <li>

                                                <a title="OEE-2" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-2" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">OEE-2</a>

                                            </li>

                                            <li>

                                                <a title="OEE-3" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-3" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">OEE-3</a>

                                            </li>

                                            <li>

                                                <a title="OEE-4" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-4" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">OEE-4</a>

                                            </li>

                                            <li>

                                                <a title="Global Overview" href="https://www.unwomen.org/en/executive-board/strategic-plan/global-overview" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Global Overview</a>

                                            </li>

                                            <li>

                                                <a title="Resources" href="https://www.unwomen.org/en/executive-board/strategic-plan/resources" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Resources</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="Calendar" href="https://www.unwomen.org/en/executive-board/calendar" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Calendar</a>



                            </li>

                            <li>



                                <a title="Membership" href="https://www.unwomen.org/en/executive-board/members" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Membership</a>



                            </li>

                            <li>



                                <a title="Bureau" href="https://www.unwomen.org/en/executive-board/bureau" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Bureau</a>



                            </li>

                            <li>



                                <a title="Secretariat" href="https://www.unwomen.org/en/executive-board/secretary" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Secretariat</a>



                            </li>

                            <li>



                                <a title="Session documents" href="https://www.unwomen.org/en/executive-board/documents" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Session documents</a>



                                    <ul>

                                            <li>

                                                <a title="2020" href="https://www.unwomen.org/en/executive-board/documents/2020" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2020</a>

                                            </li>

                                            <li>

                                                <a title="2019 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2019" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2019 sessions and other meetings</a>

                                            </li>

                                            <li>

                                                <a title="2018 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2018" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2018 sessions and other meetings</a>

                                            </li>

                                            <li>

                                                <a title="2017 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2017" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2017 sessions and other meetings</a>

                                            </li>

                                            <li>

                                                <a title="2016 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2016" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2016 sessions and other meetings</a>

                                            </li>

                                            <li>

                                                <a title="2015 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2015" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">2015 sessions and other meetings</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="Compendiums of decisions" href="https://www.unwomen.org/en/executive-board/decisions" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Compendiums of decisions</a>



                            </li>

                            <li>



                                <a title="Reports of sessions" href="https://www.unwomen.org/en/executive-board/reports" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Reports of sessions</a>



                            </li>

                            <li>



                                <a title="Key Documents" href="https://www.unwomen.org/en/executive-board/key-documents" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Key Documents</a>



                            </li>

                            <li>



                                <a title="Useful Links" href="https://www.unwomen.org/en/executive-board/useful-links" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Useful Links</a>



                            </li>

                    </ul>

            </li>

            <li>

                <a class="multi-line&#32;ga-event" data-action="ui-topnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/csw" title="Commission&#32;on&#32;the&#32;Status&#32;of&#32;Women">Commission on<span class="top-nevigation-menu-wrap"> the Status of Women</span></a>

                



                    <ul class="topdropToLeft">

                            <li>



                                <a title="Brief history" href="https://www.unwomen.org/en/csw/brief-history" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Brief history</a>



                            </li>

                            <li>



                                <a title="CSW snapshot" href="https://www.unwomen.org/en/csw/csw-snapshot" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">CSW snapshot</a>



                            </li>

                            <li>



                                <a title="CSW64 / Beijing+25 (2020)" href="https://www.unwomen.org/en/csw/csw64-2020" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">CSW64 / Beijing+25 (2020)</a>



                                    <ul>

                                            <li>

                                                <a title="Preparations" href="https://www.unwomen.org/en/csw/csw64-2020/preparations" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Preparations</a>

                                            </li>

                                            <li>

                                                <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw64-2020/official-documents" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Official Documents</a>

                                            </li>

                                            <li>

                                                <a title="Side Events" href="https://www.unwomen.org/en/csw/csw64-2020/side-events" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Side Events</a>

                                            </li>

                                            <li>

                                                <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw64-2020/session-outcomes" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Session Outcomes</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="CSW63 (2019)" href="https://www.unwomen.org/en/csw/csw63-2019" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">CSW63 (2019)</a>



                                    <ul>

                                            <li>

                                                <a title="Preparations" href="https://www.unwomen.org/en/csw/csw63-2019/preparations" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Preparations</a>

                                            </li>

                                            <li>

                                                <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw63-2019/official-documents" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Official Documents</a>

                                            </li>

                                            <li>

                                                <a title="Official Meetings" href="https://www.unwomen.org/en/csw/csw63-2019/official-meetings" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Official Meetings</a>

                                            </li>

                                            <li>

                                                <a title="Side Events" href="https://www.unwomen.org/en/csw/csw63-2019/side-events" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Side Events</a>

                                            </li>

                                            <li>

                                                <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw63-2019/session-outcomes" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Session Outcomes</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="CSW62 (2018)" href="https://www.unwomen.org/en/csw/csw62-2018" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">CSW62 (2018)</a>



                                    <ul>

                                            <li>

                                                <a title="Preparations" href="https://www.unwomen.org/en/csw/csw62-2018/preparations" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Preparations</a>

                                            </li>

                                            <li>

                                                <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw62-2018/official-documents" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Official Documents</a>

                                            </li>

                                            <li>

                                                <a title="Official Meetings" href="https://www.unwomen.org/en/csw/csw62-2018/official-meetings" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Official Meetings</a>

                                            </li>

                                            <li>

                                                <a title="Side Events" href="https://www.unwomen.org/en/csw/csw62-2018/side-events" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Side Events</a>

                                            </li>

                                            <li>

                                                <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw62-2018/session-outcomes" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Session Outcomes</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="Previous sessions" href="https://www.unwomen.org/en/csw/previous-sessions" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Previous sessions</a>



                                    <ul>

                                            <li>

                                                <a title="CSW61 (2017)" href="https://www.unwomen.org/en/csw/previous-sessions/csw61-2017" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">CSW61 (2017)</a>

                                            </li>

                                            <li>

                                                <a title="CSW60 (2016)" href="https://www.unwomen.org/en/csw/previous-sessions/csw60-2016" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">CSW60 (2016)</a>

                                            </li>

                                            <li>

                                                <a title="CSW59 / Beijing+20 (2015)" href="https://www.unwomen.org/en/csw/previous-sessions/csw59-2015" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">CSW59 / Beijing+20 (2015)</a>

                                            </li>

                                            <li>

                                                <a title="CSW58 (2014)" href="https://www.unwomen.org/en/csw/previous-sessions/csw58-2014" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">CSW58 (2014)</a>

                                            </li>

                                            <li>

                                                <a title="CSW57 (2013)" href="https://www.unwomen.org/en/csw/previous-sessions/csw57-2013" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">CSW57 (2013)</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="Member States" href="https://www.unwomen.org/en/csw/member-states" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Member States</a>



                            </li>

                            <li>



                                <a title="NGO participation" href="https://www.unwomen.org/en/csw/ngo-participation" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">NGO participation</a>



                                    <ul>

                                            <li>

                                                <a title="Eligibility" href="https://www.unwomen.org/en/csw/ngo-participation/eligibility" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Eligibility</a>

                                            </li>

                                            <li>

                                                <a title="Registration" href="https://www.unwomen.org/en/csw/ngo-participation/registration" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Registration</a>

                                            </li>

                                            <li>

                                                <a title="Opportunities for NGOs to address the Commission" href="https://www.unwomen.org/en/csw/ngo-participation/written-and-oral-statements" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Opportunities for NGOs to address the Commission</a>

                                            </li>

                                            <li>

                                                <a title="Accessibility" href="https://www.unwomen.org/en/csw/ngo-participation/accessibility" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Accessibility</a>

                                            </li>

                                            <li>

                                                <a title="NGO advisories" href="https://www.unwomen.org/en/csw/ngo-participation/ngo-advisories" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">NGO advisories</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="Communications procedure" href="https://www.unwomen.org/en/csw/communications-procedure" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Communications procedure</a>



                            </li>

                            <li>



                                <a title="Outcomes" href="https://www.unwomen.org/en/csw/outcomes" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Outcomes</a>



                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-topnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/trust-funds" title="Trust&#32;funds">Trust funds</a>

                



                    <ul class="topdropToLeft">

                            <li>



                                <a title="Fund for Gender Equality" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Fund for Gender Equality</a>



                                    <ul>

                                            <li>

                                                <a title="Our model" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/our-model" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Our model</a>

                                            </li>

                                            <li>

                                                <a title="Grant making" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/grantmaking" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Grant making</a>

                                            </li>

                                            <li>

                                                <a title="Accompaniment and growth" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/accompaniment-and-growth" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Accompaniment and growth</a>

                                            </li>

                                            <li>

                                                <a title="Results and impact" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/results-and-impact" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Results and impact</a>

                                            </li>

                                            <li>

                                                <a title="Knowledge and learning" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/knowledge-and-learning" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Knowledge and learning</a>

                                            </li>

                                            <li>

                                                <a title="Social innovation" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/social-innovation" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Social innovation</a>

                                            </li>

                                            <li>

                                                <a title="Join us" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/join-us" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Join us</a>

                                            </li>

                                            <li>

                                                <a title="Materials" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/materials" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Materials</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>



                                <a title="UN Trust Fund to End Violence against Women" href="https://www.unwomen.org/en/trust-funds/un-trust-fund-to-end-violence-against-women" class="ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">UN Trust Fund to End Violence against Women</a>



                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-topnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/get-involved" title="Get&#32;involved">Get involved</a>

                



                    <ul class="topdropToLeft">

                            <li>



                                <a title="Generation Equality" href="https://www.unwomen.org/en/get-involved/beijing-plus-25" class="arrow ga-event" data-category="ui-nav" data-action="ui-topnav-secondary">Generation Equality</a>



                                    <ul>

                                            <li>

                                                <a title="About Generation Equality" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/about" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">About Generation Equality</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality Forum" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/generation-equality-forum" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Generation Equality Forum</a>

                                            </li>

                                            <li>

                                                <a title="Action packs" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/take-action" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Action packs</a>

                                            </li>

                                            <li>

                                                <a title="Toolkit" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/toolkit" class="ga-event" data-category="ui-nav" data-action="ui-topnav-tertiary">Toolkit</a>

                                            </li>

                                    </ul>

                            </li>

                    </ul>

            </li>

    </ul>

</div>







</div>



<!-- Header ends -->



    <div role="main">





<div style="clear: both"></div>

<!--Menu For Small Device-->

<div class="column demo-3">

    <div id="dl-menu" class="dl-menuwrapper">

        <button class="dl-trigger" style="float: left;">Open Menu</button>

        <div style="clear: both;"></div>

        <ul class="dl-menu">

                <li>

                    <a title="About" href="https://www.unwomen.org/en/about-us">About</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="About" href="https://www.unwomen.org/en/about-us">About</a>

                            </li>

                                <li>

                                    <a title="About UN Women" href="https://www.unwomen.org/en/about-us/about-un-women">About UN Women</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="About UN Women" href="https://www.unwomen.org/en/about-us/about-un-women">About UN Women</a>

                                            </li>

                                                <li>

                                                    <a title="&#8216;One Woman&#8217; &#8211; The UN Women song" href="https://www.unwomen.org/en/about-us/about-un-women/un-women-song">‘One Woman’ – The UN Women song</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Directorate" href="https://www.unwomen.org/en/about-us/directorate">Directorate</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Directorate" href="https://www.unwomen.org/en/about-us/directorate">Directorate</a>

                                            </li>

                                                <li>

                                                    <a title="Executive Director" href="https://www.unwomen.org/en/about-us/directorate/executive-director">Executive Director</a>

                                                </li>

                                                <li>

                                                    <a title="Deputy Executive Director for Normative Support, UN System Coordination and Programme Results" href="https://www.unwomen.org/en/about-us/directorate/ded-normative-support-un-system-coordination-and-programme-results">Deputy Executive Director for Normative Support, UN System Coordination and Programme Results</a>

                                                </li>

                                                <li>

                                                    <a title="Deputy Executive Director for Resource Management, Sustainability and Partnerships" href="https://www.unwomen.org/en/about-us/directorate/ded-resource-management-sustainability-and-partnerships">Deputy Executive Director for Resource Management, Sustainability and Partnerships</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Governance" href="https://www.unwomen.org/en/about-us/governance">Governance</a>

                                </li>

                                <li>

                                    <a title="Guiding documents" href="https://www.unwomen.org/en/about-us/guiding-documents">Guiding documents</a>

                                </li>

                                <li>

                                    <a title="Accountability" href="https://www.unwomen.org/en/about-us/accountability">Accountability</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Accountability" href="https://www.unwomen.org/en/about-us/accountability">Accountability</a>

                                            </li>

                                                <li>

                                                    <a title="Evaluation" href="https://www.unwomen.org/en/about-us/accountability/evaluation">Evaluation</a>

                                                </li>

                                                <li>

                                                    <a title="Audit" href="https://www.unwomen.org/en/about-us/accountability/audit">Audit</a>

                                                </li>

                                                <li>

                                                    <a title="Report wrongdoing" href="https://www.unwomen.org/en/about-us/accountability/investigations">Report wrongdoing</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Employment" href="https://www.unwomen.org/en/about-us/employment">Employment</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Employment" href="https://www.unwomen.org/en/about-us/employment">Employment</a>

                                            </li>

                                                <li>

                                                    <a title="Internship programme" href="https://www.unwomen.org/en/about-us/employment/internship-programme">Internship programme</a>

                                                </li>

                                                <li>

                                                    <a title="UN Women Alumni Association" href="https://www.unwomen.org/en/about-us/employment/un-women-alumni-association">UN Women Alumni Association</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Procurement" href="https://www.unwomen.org/en/about-us/procurement">Procurement</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Procurement" href="https://www.unwomen.org/en/about-us/procurement">Procurement</a>

                                            </li>

                                                <li>

                                                    <a title="Procurement principles" href="https://www.unwomen.org/en/about-us/procurement/procurement-principles">Procurement principles</a>

                                                </li>

                                                <li>

                                                    <a title="Gender-responsive procurement" href="https://www.unwomen.org/en/about-us/procurement/gender-responsive-procurement">Gender-responsive procurement</a>

                                                </li>

                                                <li>

                                                    <a title="Doing business with UN Women" href="https://www.unwomen.org/en/about-us/procurement/doing-business-with-un-women">Doing business with UN Women</a>

                                                </li>

                                                <li>

                                                    <a title="Becoming a UN Women vendor" href="https://www.unwomen.org/en/about-us/procurement/how-to-become-a-un-women-supplier">Becoming a UN Women vendor</a>

                                                </li>

                                                <li>

                                                    <a title="Contract templates and general conditions of contract" href="https://www.unwomen.org/en/about-us/procurement/contract-templates-and-general-conditions-of-contract">Contract templates and general conditions of contract</a>

                                                </li>

                                                <li>

                                                    <a title="Vendor protest procedure" href="https://www.unwomen.org/en/about-us/procurement/vendor-protest-procedure">Vendor protest procedure</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Contact us" href="https://www.unwomen.org/en/about-us/contact-us">Contact us</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="What we&amp;nbsp;do" href="https://www.unwomen.org/en/what-we-do">What we&nbsp;do</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="What we&amp;nbsp;do" href="https://www.unwomen.org/en/what-we-do">What we&nbsp;do</a>

                            </li>

                                <li>

                                    <a title="Leadership and political participation" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation">Leadership and political participation</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Leadership and political participation" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation">Leadership and political participation</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Women&#8217;s movements" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/womens-movements">Women’s movements</a>

                                                </li>

                                                <li>

                                                    <a title="Parliaments and local governance" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/parliaments-and-local-governance">Parliaments and local governance</a>

                                                </li>

                                                <li>

                                                    <a title="Constitutions and legal reform" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/constitutions-and-legal-reform">Constitutions and legal reform</a>

                                                </li>

                                                <li>

                                                    <a title="Elections" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/elections">Elections</a>

                                                </li>

                                                <li>

                                                    <a title="Media" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/media">Media</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Economic empowerment" href="https://www.unwomen.org/en/what-we-do/economic-empowerment">Economic empowerment</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Economic empowerment" href="https://www.unwomen.org/en/what-we-do/economic-empowerment">Economic empowerment</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and Figures" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/facts-and-figures">Facts and Figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global Norms and Standards" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/global-norms-and-standards">Global Norms and Standards</a>

                                                </li>

                                                <li>

                                                    <a title="Macroeconomic policies and social protection" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/macroeconomics-policies-and-social-protection">Macroeconomic policies and social protection</a>

                                                </li>

                                                <li>

                                                    <a title="Sustainable Development and Climate Change" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/sustainable-development-and-climate-change">Sustainable Development and Climate Change</a>

                                                </li>

                                                <li>

                                                    <a title="Rural women" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/rural-women">Rural women</a>

                                                </li>

                                                <li>

                                                    <a title="Employment and migration" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/employment-and-migration">Employment and migration</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Ending violence against women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women">Ending violence against women</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Ending violence against women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women">Ending violence against women</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="FAQs" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/faqs">FAQs</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Passing and implementing effective laws and policies" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/passing-strong-laws-and-policies">Passing and implementing effective laws and policies</a>

                                                </li>

                                                <li>

                                                    <a title="Services for all women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/services-for-all-women">Services for all women</a>

                                                </li>

                                                <li>

                                                    <a title="Increasing knowledge and awareness" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/increasing-knowledge-and-awareness">Increasing knowledge and awareness</a>

                                                </li>

                                                <li>

                                                    <a title="Creating safe public spaces" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/creating-safe-public-spaces">Creating safe public spaces</a>

                                                </li>

                                                <li>

                                                    <a title="Focusing on prevention to stop the violence" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/prevention">Focusing on prevention to stop the violence</a>

                                                </li>

                                                <li>

                                                    <a title="UNiTE campaign" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/take-action">UNiTE campaign</a>

                                                </li>

                                                <li>

                                                    <a title="Spokesperson on Addressing Sexual Harassment" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/spokesperson-on-addressing-sexual-harassment">Spokesperson on Addressing Sexual Harassment</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security">Peace and security</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security">Peace and security</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/peace-and-security/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/peace-and-security/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Conflict prevention and resolution" href="https://www.unwomen.org/en/what-we-do/peace-and-security/conflict-prevention-and-resolution">Conflict prevention and resolution</a>

                                                </li>

                                                <li>

                                                    <a title="Building and sustaining peace" href="https://www.unwomen.org/en/what-we-do/peace-and-security/building-and-sustaining-peace">Building and sustaining peace</a>

                                                </li>

                                                <li>

                                                    <a title="Young women in peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security/young-women-in-peace-and-security">Young women in peace and security</a>

                                                </li>

                                                <li>

                                                    <a title="Rule of law: Justice and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security/rule-of-law-and-justice">Rule of law: Justice and security</a>

                                                </li>

                                                <li>

                                                    <a title="Women, peace, and security in the work of the UN Security Council" href="https://www.unwomen.org/en/what-we-do/peace-and-security/un-security-council">Women, peace, and security in the work of the UN Security Council</a>

                                                </li>

                                                <li>

                                                    <a title="Preventing violent extremism and countering terrorism" href="https://www.unwomen.org/en/what-we-do/peace-and-security/preventing-violent-extremism">Preventing violent extremism and countering terrorism</a>

                                                </li>

                                                <li>

                                                    <a title="Planning and monitoring" href="https://www.unwomen.org/en/what-we-do/peace-and-security/planning-and-monitoring">Planning and monitoring</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Humanitarian action" href="https://www.unwomen.org/en/what-we-do/humanitarian-action">Humanitarian action</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Humanitarian action" href="https://www.unwomen.org/en/what-we-do/humanitarian-action">Humanitarian action</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Humanitarian coordination" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/humanitarian-coordination">Humanitarian coordination</a>

                                                </li>

                                                <li>

                                                    <a title="Crisis response and recovery " href="https://www.unwomen.org/en/what-we-do/humanitarian-action/emergency-response">Crisis response and recovery </a>

                                                </li>

                                                <li>

                                                    <a title="Disaster risk reduction" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/disaster-risk-reduction">Disaster risk reduction</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Youth " href="https://www.unwomen.org/en/what-we-do/youth">Youth </a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Youth " href="https://www.unwomen.org/en/what-we-do/youth">Youth </a>

                                            </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/youth/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Strengthening young women&#39;s leadership" href="https://www.unwomen.org/en/what-we-do/youth/strengthening-young-women-leadership">Strengthening young women's leadership</a>

                                                </li>

                                                <li>

                                                    <a title="Economic empowerment and skills development for young women" href="https://www.unwomen.org/en/what-we-do/youth/economic-empowerment-and-skills-development-for-young-women">Economic empowerment and skills development for young women</a>

                                                </li>

                                                <li>

                                                    <a title="Action on ending violence against young women and girls" href="https://www.unwomen.org/en/what-we-do/youth/action-on-ending-violence-against-young-women-and-girls">Action on ending violence against young women and girls</a>

                                                </li>

                                                <li>

                                                    <a title="Engaging boys and young men in gender equality" href="https://www.unwomen.org/en/what-we-do/youth/engaging-boys-and-young-men-in-gender-equality">Engaging boys and young men in gender equality</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Governance and national planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning">Governance and national planning</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Governance and national planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning">Governance and national planning</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Inclusive National Planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/inclusive-national-planning">Inclusive National Planning</a>

                                                </li>

                                                <li>

                                                    <a title="Public Sector Reform" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/engaging-in-public-sector-reform">Public Sector Reform</a>

                                                </li>

                                                <li>

                                                    <a title="Tracking Investments" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/tracking-investments">Tracking Investments</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Sustainable development agenda" href="https://www.unwomen.org/en/what-we-do/post-2015">Sustainable development agenda</a>

                                </li>

                                <li>

                                    <a title="HIV and AIDS" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids">HIV and AIDS</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="HIV and AIDS" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids">HIV and AIDS</a>

                                            </li>

                                                <li>

                                                    <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/facts-and-figures">Facts and figures</a>

                                                </li>

                                                <li>

                                                    <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/global-norms-and-standards">Global norms and standards</a>

                                                </li>

                                                <li>

                                                    <a title="Leadership and Participation" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/leadership-and-participation">Leadership and Participation</a>

                                                </li>

                                                <li>

                                                    <a title="National Planning" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/national-planning">National Planning</a>

                                                </li>

                                                <li>

                                                    <a title="Violence against Women" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/violence-against-women">Violence against Women</a>

                                                </li>

                                                <li>

                                                    <a title="Access to Justice" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/access-to-justice">Access to Justice</a>

                                                </li>

                                        </ul>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Where we are" href="https://www.unwomen.org/en/where-we-are">Where we are</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Where we are" href="https://www.unwomen.org/en/where-we-are">Where we are</a>

                            </li>

                                <li>

                                    <a title="Africa" href="https://www.unwomen.org/en/where-we-are/africa">Africa</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Africa" href="https://www.unwomen.org/en/where-we-are/africa">Africa</a>

                                            </li>

                                                <li>

                                                    <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/africa/regional-and-country-offices">Regional and country offices</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Americas and the Caribbean" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean">Americas and the Caribbean</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Americas and the Caribbean" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean">Americas and the Caribbean</a>

                                            </li>

                                                <li>

                                                    <a title="Regional and Country Offices" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean/regional-and-country-offices">Regional and Country Offices</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Arab States/&amp;#8203;North&amp;nbsp;Africa" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa">Arab States/&#8203;North&nbsp;Africa</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Arab States/&amp;#8203;North&amp;nbsp;Africa" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa">Arab States/&#8203;North&nbsp;Africa</a>

                                            </li>

                                                <li>

                                                    <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa/regional-and-country-offices">Regional and country offices</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Asia and the&#160;Pacific" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific">Asia and the Pacific</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Asia and the&#160;Pacific" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific">Asia and the Pacific</a>

                                            </li>

                                                <li>

                                                    <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific/regional-and-country-offices">Regional and country offices</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Europe and Central&#160;Asia" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia">Europe and Central Asia</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Europe and Central&#160;Asia" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia">Europe and Central Asia</a>

                                            </li>

                                                <li>

                                                    <a title="Regional and Country Offices" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia/regional-and-country-offices">Regional and Country Offices</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Liaison offices" href="https://www.unwomen.org/en/where-we-are/liaison-offices">Liaison offices</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="How we work" href="https://www.unwomen.org/en/how-we-work">How we work</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="How we work" href="https://www.unwomen.org/en/how-we-work">How we work</a>

                            </li>

                                <li>

                                    <a title="Flagship programme initiatives" href="https://www.unwomen.org/en/how-we-work/flagship-programmes">Flagship programme initiatives</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Flagship programme initiatives" href="https://www.unwomen.org/en/how-we-work/flagship-programmes">Flagship programme initiatives</a>

                                            </li>

                                                <li>

                                                    <a title="Flagship programme: Making Every Woman and Girl Count " href="https://www.unwomen.org/en/how-we-work/flagship-programmes/making-every-woman-and-girl-count">Flagship programme: Making Every Woman and Girl Count </a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Programme implementation" href="https://www.unwomen.org/en/how-we-work/programme-implementation">Programme implementation</a>

                                </li>

                                <li>

                                    <a title="Innovation and technology" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology">Innovation and technology</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Innovation and technology" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology">Innovation and technology</a>

                                            </li>

                                                <li>

                                                    <a title="UN Women Global Innovation Coalition for Change" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology/un-women-global-innovation-coalition-for-change">UN Women Global Innovation Coalition for Change</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Intergovernmental support" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support">Intergovernmental support</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Intergovernmental support" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support">Intergovernmental support</a>

                                            </li>

                                                <li>

                                                    <a title="Commission on the Status of Women" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/commission-on-the-status-of-women">Commission on the Status of Women</a>

                                                </li>

                                                <li>

                                                    <a title="Climate change and the environment" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/climate-change-and-the-environment">Climate change and the environment</a>

                                                </li>

                                                <li>

                                                    <a title="High-Level Political Forum on Sustainable Development" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/hlpf-on-sustainable-development">High-Level Political Forum on Sustainable Development</a>

                                                </li>

                                                <li>

                                                    <a title="Major resolutions" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/major-resolutions">Major resolutions</a>

                                                </li>

                                                <li>

                                                    <a title="Other Intergovernmental Processes" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/other-intergovernmental-processes">Other Intergovernmental Processes</a>

                                                </li>

                                                <li>

                                                    <a title="World Conferences on Women" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/world-conferences-on-women">World Conferences on Women</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="UN system coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination">UN system coordination</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="UN system coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination">UN system coordination</a>

                                            </li>

                                                <li>

                                                    <a title="Global Coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/global-coordination">Global Coordination</a>

                                                </li>

                                                <li>

                                                    <a title="Regional and country coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/regional-and-country-coordination">Regional and country coordination</a>

                                                </li>

                                                <li>

                                                    <a title="Gender Mainstreaming" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/gender-mainstreaming">Gender Mainstreaming</a>

                                                </li>

                                                <li>

                                                    <a title="Promoting UN accountability" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/promoting-un-accountability">Promoting UN accountability</a>

                                                </li>

                                                <li>

                                                    <a title="Coordination resources" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/coordination-resources">Coordination resources</a>

                                                </li>

                                                <li>

                                                    <a title="UN Coordination Library" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/library">UN Coordination Library</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Gender parity in the United Nations" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations">Gender parity in the United Nations</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Gender parity in the United Nations" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations">Gender parity in the United Nations</a>

                                            </li>

                                                <li>

                                                    <a title="System-wide strategy" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/system-wide-strategy">System-wide strategy</a>

                                                </li>

                                                <li>

                                                    <a title="Gender Focal Points and Focal Points for Women" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/focal-points-for-women">Gender Focal Points and Focal Points for Women</a>

                                                </li>

                                                <li>

                                                    <a title="Data and statistics" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/current-status-of-women">Data and statistics</a>

                                                </li>

                                                <li>

                                                    <a title="Laws and policies" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/laws-and-policies">Laws and policies</a>

                                                </li>

                                                <li>

                                                    <a title="Strategies and tools" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/strategies-and-tools">Strategies and tools</a>

                                                </li>

                                                <li>

                                                    <a title="Reports and monitoring" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/reports-and-monitoring">Reports and monitoring</a>

                                                </li>

                                                <li>

                                                    <a title="Other resources" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/other-resources">Other resources</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Training for gender equality and women&#39;s empowerment" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training">Training for gender equality and women's empowerment</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Training for gender equality and women&#39;s empowerment" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training">Training for gender equality and women's empowerment</a>

                                            </li>

                                                <li>

                                                    <a title="Training Centre services" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training/training-centre-services">Training Centre services</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Research and data" href="https://www.unwomen.org/en/how-we-work/research-and-data">Research and data</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Research and data" href="https://www.unwomen.org/en/how-we-work/research-and-data">Research and data</a>

                                            </li>

                                                <li>

                                                    <a title="Publications" href="https://www.unwomen.org/en/how-we-work/research-and-data/publications">Publications</a>

                                                </li>

                                        </ul>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Partnerships" href="https://www.unwomen.org/en/partnerships">Partnerships</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Partnerships" href="https://www.unwomen.org/en/partnerships">Partnerships</a>

                            </li>

                                <li>

                                    <a title="Government partners" href="https://www.unwomen.org/en/partnerships/donor-countries">Government partners</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Government partners" href="https://www.unwomen.org/en/partnerships/donor-countries">Government partners</a>

                                            </li>

                                                <li>

                                                    <a title="Top government partners" href="https://www.unwomen.org/en/partnerships/donor-countries/top-donors">Top government partners</a>

                                                </li>

                                                <li>

                                                    <a title="Core resources" href="https://www.unwomen.org/en/partnerships/donor-countries/core-resources">Core resources</a>

                                                </li>

                                                <li>

                                                    <a title="Non-core resources" href="https://www.unwomen.org/en/partnerships/donor-countries/non-core-resources">Non-core resources</a>

                                                </li>

                                                <li>

                                                    <a title="Contribution trends" href="https://www.unwomen.org/en/partnerships/donor-countries/contribution-trends">Contribution trends</a>

                                                </li>

                                                <li>

                                                    <a title="Frequently asked questions" href="https://www.unwomen.org/en/partnerships/donor-countries/frequently-asked-questions">Frequently asked questions</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="National mechanisms" href="https://www.unwomen.org/en/partnerships/national-mechanisms">National mechanisms</a>

                                </li>

                                <li>

                                    <a title="Civil society" href="https://www.unwomen.org/en/partnerships/civil-society">Civil society</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Civil society" href="https://www.unwomen.org/en/partnerships/civil-society">Civil society</a>

                                            </li>

                                                <li>

                                                    <a title="Civil Society Advisory Groups" href="https://www.unwomen.org/en/partnerships/civil-society/civil-society-advisory-groups">Civil Society Advisory Groups</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Businesses and philanthropies" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations">Businesses and philanthropies</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Businesses and philanthropies" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations">Businesses and philanthropies</a>

                                            </li>

                                                <li>

                                                    <a title="Benefits of partnering with UN&amp;nbsp;Women" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations/why-un-women">Benefits of partnering with UN&nbsp;Women</a>

                                                </li>

                                                <li>

                                                    <a title="Business and philanthropic partners" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations/major-partners">Business and philanthropic partners</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="National Committees" href="https://www.unwomen.org/en/partnerships/national-committees">National Committees</a>

                                </li>

                                <li>

                                    <a title="Goodwill Ambassadors" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors">Goodwill Ambassadors</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Goodwill Ambassadors" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors">Goodwill Ambassadors</a>

                                            </li>

                                                <li>

                                                    <a title="Danai Gurira" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/danai-gurira">Danai Gurira</a>

                                                </li>

                                                <li>

                                                    <a title="Nicole Kidman" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/nicole-kidman">Nicole Kidman</a>

                                                </li>

                                                <li>

                                                    <a title="Anne Hathaway" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/anne-hathaway">Anne Hathaway</a>

                                                </li>

                                                <li>

                                                    <a title="Marta Vieira da Silva" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/marta-vieira-da-silva">Marta Vieira da Silva</a>

                                                </li>

                                                <li>

                                                    <a title="Emma Watson" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/emma-watson">Emma Watson</a>

                                                </li>

                                                <li>

                                                    <a title="Farhan Akhtar" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/farhan-akhtar">Farhan Akhtar</a>

                                                </li>

                                                <li>

                                                    <a title="Princess Bajrakitiyabha Mahidol" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/princess-bajrakitiyabha">Princess Bajrakitiyabha Mahidol</a>

                                                </li>

                                                <li>

                                                    <a title="Tong Dawei" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/tong-dawei">Tong Dawei</a>

                                                </li>

                                                <li>

                                                    <a title="Jaha Dukureh" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/jaha-dukureh">Jaha Dukureh</a>

                                                </li>

                                                <li>

                                                    <a title="Muniba Mazari" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/muniba-mazari">Muniba Mazari</a>

                                                </li>

                                                <li>

                                                    <a title="Sania Mirza" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/sania-mirza">Sania Mirza</a>

                                                </li>

                                                <li>

                                                    <a title="Camila Pitanga" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/camila-pitanga">Camila Pitanga</a>

                                                </li>

                                                <li>

                                                    <a title="Hai Qing" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/hai-qing">Hai Qing</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Media collaboration" href="https://www.unwomen.org/en/partnerships/media-collaboration">Media collaboration</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Media collaboration" href="https://www.unwomen.org/en/partnerships/media-collaboration">Media collaboration</a>

                                            </li>

                                                <li>

                                                    <a title="UN Women Media Compact" href="https://www.unwomen.org/en/partnerships/media-collaboration/media-compact">UN Women Media Compact</a>

                                                </li>

                                        </ul>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="News and events" href="https://www.unwomen.org/en/news">News and events</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="News and events" href="https://www.unwomen.org/en/news">News and events</a>

                            </li>

                                <li>

                                    <a title="News" href="https://www.unwomen.org/en/news/stories">News</a>

                                </li>

                                <li>

                                    <a title="Editorial series" href="https://www.unwomen.org/en/news/editorial-series">Editorial series</a>

                                </li>

                                <li>

                                    <a title="In Focus" href="https://www.unwomen.org/en/news/in-focus">In Focus</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="In Focus" href="https://www.unwomen.org/en/news/in-focus">In Focus</a>

                                            </li>

                                                <li>

                                                    <a title="Gender equality matters in COVID-19 response" href="https://www.unwomen.org/en/news/in-focus/in-focus-gender-equality-in-covid-19-response">Gender equality matters in COVID-19 response</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality Action Pack, March 2020" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-march-2020">Generation Equality Action Pack, March 2020</a>

                                                </li>

                                                <li>

                                                    <a title="International Women&#8217;s Day" href="https://www.unwomen.org/en/news/in-focus/international-womens-day">International Women’s Day</a>

                                                </li>

                                                <li>

                                                    <a title="International Day of Women and Girls in Science" href="https://www.unwomen.org/en/news/in-focus/international-day-of-women-and-girls-in-science">International Day of Women and Girls in Science</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality action pack, January 2020" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-january-2020">Generation Equality action pack, January 2020</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality action pack, December 2019" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-december-2019">Generation Equality action pack, December 2019</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality action pack, November 2019: Generation Equality Stands against Rape" href="https://www.unwomen.org/en/news/in-focus/end-violence-against-women">Generation Equality action pack, November 2019: Generation Equality Stands against Rape</a>

                                                </li>

                                                <li>

                                                    <a title="Women, peace and security" href="https://www.unwomen.org/en/news/in-focus/women-peace-security">Women, peace and security</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality action pack, October 2019" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-october-2019">Generation Equality action pack, October 2019</a>

                                                </li>

                                                <li>

                                                    <a title="International Day of Rural Women" href="https://www.unwomen.org/en/news/in-focus/rural-women-day">International Day of Rural Women</a>

                                                </li>

                                                <li>

                                                    <a title="International Day of the Girl Child" href="https://www.unwomen.org/en/news/in-focus/girl-child">International Day of the Girl Child</a>

                                                </li>

                                                <li>

                                                    <a title="74th session of the UN General Assembly" href="https://www.unwomen.org/en/news/in-focus/general-assembly">74th session of the UN General Assembly</a>

                                                </li>

                                                <li>

                                                    <a title="World  Humanitarian Day" href="https://www.unwomen.org/en/news/in-focus/humanitarian-action">World  Humanitarian Day</a>

                                                </li>

                                                <li>

                                                    <a title=" International Youth Day" href="https://www.unwomen.org/en/news/in-focus/youth"> International Youth Day</a>

                                                </li>

                                                <li>

                                                    <a title="International Day of the World&#8217;s Indigenous Peoples " href="https://www.unwomen.org/en/news/in-focus/indigenous-women">International Day of the World’s Indigenous Peoples </a>

                                                </li>

                                                <li>

                                                    <a title="World Refugee Day" href="https://www.unwomen.org/en/news/in-focus/world-refugee-day">World Refugee Day</a>

                                                </li>

                                                <li>

                                                    <a title="Women and girls in sport" href="https://www.unwomen.org/en/news/in-focus/women-and-sport">Women and girls in sport</a>

                                                </li>

                                                <li>

                                                    <a title="International Girls in ICT Day" href="https://www.unwomen.org/en/news/in-focus/international-girls-in-ict-day">International Girls in ICT Day</a>

                                                </li>

                                                <li>

                                                    <a title="CSW63" href="https://www.unwomen.org/en/news/in-focus/csw">CSW63</a>

                                                </li>

                                                <li>

                                                    <a title="CSW62" href="https://www.unwomen.org/en/news/in-focus/csw62">CSW62</a>

                                                </li>

                                                <li>

                                                    <a title="Women and the SDGs" href="https://www.unwomen.org/en/news/in-focus/women-and-the-sdgs">Women and the SDGs</a>

                                                </li>

                                                <li>

                                                    <a title="In Focus: Climate action by, and for, women" href="https://www.unwomen.org/en/news/in-focus/climate-change">In Focus: Climate action by, and for, women</a>

                                                </li>

                                                <li>

                                                    <a title="Indigenous women&#8217;s rights and activism" href="https://www.unwomen.org/en/news/in-focus/indigenous-womens-rights-and-activism">Indigenous women’s rights and activism</a>

                                                </li>

                                                <li>

                                                    <a title="Empowering women to conserve our oceans" href="https://www.unwomen.org/en/news/in-focus/empowering-women-to-conserve-our-oceans">Empowering women to conserve our oceans</a>

                                                </li>

                                                <li>

                                                    <a title="CSW61" href="https://www.unwomen.org/en/news/in-focus/csw61">CSW61</a>

                                                </li>

                                                <li>

                                                    <a title="Women refugees and migrants" href="https://www.unwomen.org/en/news/in-focus/women-refugees-and-migrants">Women refugees and migrants</a>

                                                </li>

                                                <li>

                                                    <a title="CSW60" href="https://www.unwomen.org/en/news/in-focus/csw60">CSW60</a>

                                                </li>

                                                <li>

                                                    <a title="Financing: Why it matters for women and girls" href="https://www.unwomen.org/en/news/in-focus/financing-for-gender-equality">Financing: Why it matters for women and girls</a>

                                                </li>

                                                <li>

                                                    <a title="Engaging Men" href="https://www.unwomen.org/en/news/in-focus/engaging-men">Engaging Men</a>

                                                </li>

                                                <li>

                                                    <a title="SIDS Conference" href="https://www.unwomen.org/en/news/in-focus/sids">SIDS Conference</a>

                                                </li>

                                                <li>

                                                    <a title="MDG Momentum" href="https://www.unwomen.org/en/news/in-focus/mdg-momentum">MDG Momentum</a>

                                                </li>

                                                <li>

                                                    <a title="Strengthening Women&#8217;s Access to Justice" href="https://www.unwomen.org/en/news/in-focus/strengthening-womens-access-to-justice">Strengthening Women’s Access to Justice</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Events" href="https://www.unwomen.org/en/news/events">Events</a>

                                </li>

                                <li>

                                    <a title="Media contacts" href="https://www.unwomen.org/en/news/media-contacts">Media contacts</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Digital library" href="https://www.unwomen.org/en/digital-library">Digital library</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Digital library" href="https://www.unwomen.org/en/digital-library">Digital library</a>

                            </li>

                                <li>

                                    <a title="Publications" href="https://www.unwomen.org/en/digital-library/publications">Publications</a>

                                </li>

                                <li>

                                    <a title="Multimedia" href="https://www.unwomen.org/en/digital-library/multimedia">Multimedia</a>

                                </li>

                                <li>

                                    <a title="Annual report" href="https://www.unwomen.org/en/digital-library/annual-report">Annual report</a>

                                </li>

                                <li>

                                    <a title="SDG monitoring report" href="https://www.unwomen.org/en/digital-library/sdg-report">SDG monitoring report</a>

                                </li>

                                <li>

                                    <a title="Progress of the world&#8217;s women" href="https://www.unwomen.org/en/digital-library/progress-of-the-worlds-women">Progress of the world’s women</a>

                                </li>

                                <li>

                                    <a title="World survey on the role of women in development" href="https://www.unwomen.org/en/digital-library/world-survey-on-the-role-of-women-in-development">World survey on the role of women in development</a>

                                </li>

                                <li>

                                    <a title="Reprint permissions" href="https://www.unwomen.org/en/digital-library/reprint-permissions">Reprint permissions</a>

                                </li>

                                <li>

                                    <a title="GenderTerm" href="https://www.unwomen.org/en/digital-library/genderterm">GenderTerm</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Executive Board" href="https://www.unwomen.org/en/executive-board">Executive Board</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Executive Board" href="https://www.unwomen.org/en/executive-board">Executive Board</a>

                            </li>

                                <li>

                                    <a title="Strategic Plan" href="https://www.unwomen.org/en/executive-board/strategic-plan">Strategic Plan</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Strategic Plan" href="https://www.unwomen.org/en/executive-board/strategic-plan">Strategic Plan</a>

                                            </li>

                                                <li>

                                                    <a title="Impact Area" href="https://www.unwomen.org/en/executive-board/strategic-plan/impact-area">Impact Area</a>

                                                </li>

                                                <li>

                                                    <a title="Outcome Area 1" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-1">Outcome Area 1</a>

                                                </li>

                                                <li>

                                                    <a title="Outcome Area 2" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-2">Outcome Area 2</a>

                                                </li>

                                                <li>

                                                    <a title="Outcome Area 3" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-3">Outcome Area 3</a>

                                                </li>

                                                <li>

                                                    <a title="Outcome Area 4" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-4">Outcome Area 4</a>

                                                </li>

                                                <li>

                                                    <a title="Outcome Area 5" href="https://www.unwomen.org/en/executive-board/strategic-plan/outcome-area-5">Outcome Area 5</a>

                                                </li>

                                                <li>

                                                    <a title="OEE-1" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-1">OEE-1</a>

                                                </li>

                                                <li>

                                                    <a title="OEE-2" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-2">OEE-2</a>

                                                </li>

                                                <li>

                                                    <a title="OEE-3" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-3">OEE-3</a>

                                                </li>

                                                <li>

                                                    <a title="OEE-4" href="https://www.unwomen.org/en/executive-board/strategic-plan/oee-4">OEE-4</a>

                                                </li>

                                                <li>

                                                    <a title="Global Overview" href="https://www.unwomen.org/en/executive-board/strategic-plan/global-overview">Global Overview</a>

                                                </li>

                                                <li>

                                                    <a title="Resources" href="https://www.unwomen.org/en/executive-board/strategic-plan/resources">Resources</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Calendar" href="https://www.unwomen.org/en/executive-board/calendar">Calendar</a>

                                </li>

                                <li>

                                    <a title="Membership" href="https://www.unwomen.org/en/executive-board/members">Membership</a>

                                </li>

                                <li>

                                    <a title="Bureau" href="https://www.unwomen.org/en/executive-board/bureau">Bureau</a>

                                </li>

                                <li>

                                    <a title="Secretariat" href="https://www.unwomen.org/en/executive-board/secretary">Secretariat</a>

                                </li>

                                <li>

                                    <a title="Session documents" href="https://www.unwomen.org/en/executive-board/documents">Session documents</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Session documents" href="https://www.unwomen.org/en/executive-board/documents">Session documents</a>

                                            </li>

                                                <li>

                                                    <a title="2020" href="https://www.unwomen.org/en/executive-board/documents/2020">2020</a>

                                                </li>

                                                <li>

                                                    <a title="2019 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2019">2019 sessions and other meetings</a>

                                                </li>

                                                <li>

                                                    <a title="2018 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2018">2018 sessions and other meetings</a>

                                                </li>

                                                <li>

                                                    <a title="2017 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2017">2017 sessions and other meetings</a>

                                                </li>

                                                <li>

                                                    <a title="2016 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2016">2016 sessions and other meetings</a>

                                                </li>

                                                <li>

                                                    <a title="2015 sessions and other meetings" href="https://www.unwomen.org/en/executive-board/documents/2015">2015 sessions and other meetings</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Compendiums of decisions" href="https://www.unwomen.org/en/executive-board/decisions">Compendiums of decisions</a>

                                </li>

                                <li>

                                    <a title="Reports of sessions" href="https://www.unwomen.org/en/executive-board/reports">Reports of sessions</a>

                                </li>

                                <li>

                                    <a title="Key Documents" href="https://www.unwomen.org/en/executive-board/key-documents">Key Documents</a>

                                </li>

                                <li>

                                    <a title="Useful Links" href="https://www.unwomen.org/en/executive-board/useful-links">Useful Links</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Commission on the Status of Women" href="https://www.unwomen.org/en/csw">Commission on the Status of Women</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Commission on the Status of Women" href="https://www.unwomen.org/en/csw">Commission on the Status of Women</a>

                            </li>

                                <li>

                                    <a title="Brief history" href="https://www.unwomen.org/en/csw/brief-history">Brief history</a>

                                </li>

                                <li>

                                    <a title="CSW snapshot" href="https://www.unwomen.org/en/csw/csw-snapshot">CSW snapshot</a>

                                </li>

                                <li>

                                    <a title="CSW64 / Beijing+25 (2020)" href="https://www.unwomen.org/en/csw/csw64-2020">CSW64 / Beijing+25 (2020)</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="CSW64 / Beijing+25 (2020)" href="https://www.unwomen.org/en/csw/csw64-2020">CSW64 / Beijing+25 (2020)</a>

                                            </li>

                                                <li>

                                                    <a title="Preparations" href="https://www.unwomen.org/en/csw/csw64-2020/preparations">Preparations</a>

                                                </li>

                                                <li>

                                                    <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw64-2020/official-documents">Official Documents</a>

                                                </li>

                                                <li>

                                                    <a title="Side Events" href="https://www.unwomen.org/en/csw/csw64-2020/side-events">Side Events</a>

                                                </li>

                                                <li>

                                                    <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw64-2020/session-outcomes">Session Outcomes</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="CSW63 (2019)" href="https://www.unwomen.org/en/csw/csw63-2019">CSW63 (2019)</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="CSW63 (2019)" href="https://www.unwomen.org/en/csw/csw63-2019">CSW63 (2019)</a>

                                            </li>

                                                <li>

                                                    <a title="Preparations" href="https://www.unwomen.org/en/csw/csw63-2019/preparations">Preparations</a>

                                                </li>

                                                <li>

                                                    <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw63-2019/official-documents">Official Documents</a>

                                                </li>

                                                <li>

                                                    <a title="Official Meetings" href="https://www.unwomen.org/en/csw/csw63-2019/official-meetings">Official Meetings</a>

                                                </li>

                                                <li>

                                                    <a title="Side Events" href="https://www.unwomen.org/en/csw/csw63-2019/side-events">Side Events</a>

                                                </li>

                                                <li>

                                                    <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw63-2019/session-outcomes">Session Outcomes</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="CSW62 (2018)" href="https://www.unwomen.org/en/csw/csw62-2018">CSW62 (2018)</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="CSW62 (2018)" href="https://www.unwomen.org/en/csw/csw62-2018">CSW62 (2018)</a>

                                            </li>

                                                <li>

                                                    <a title="Preparations" href="https://www.unwomen.org/en/csw/csw62-2018/preparations">Preparations</a>

                                                </li>

                                                <li>

                                                    <a title="Official Documents" href="https://www.unwomen.org/en/csw/csw62-2018/official-documents">Official Documents</a>

                                                </li>

                                                <li>

                                                    <a title="Official Meetings" href="https://www.unwomen.org/en/csw/csw62-2018/official-meetings">Official Meetings</a>

                                                </li>

                                                <li>

                                                    <a title="Side Events" href="https://www.unwomen.org/en/csw/csw62-2018/side-events">Side Events</a>

                                                </li>

                                                <li>

                                                    <a title="Session Outcomes" href="https://www.unwomen.org/en/csw/csw62-2018/session-outcomes">Session Outcomes</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Previous sessions" href="https://www.unwomen.org/en/csw/previous-sessions">Previous sessions</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Previous sessions" href="https://www.unwomen.org/en/csw/previous-sessions">Previous sessions</a>

                                            </li>

                                                <li>

                                                    <a title="CSW61 (2017)" href="https://www.unwomen.org/en/csw/previous-sessions/csw61-2017">CSW61 (2017)</a>

                                                </li>

                                                <li>

                                                    <a title="CSW60 (2016)" href="https://www.unwomen.org/en/csw/previous-sessions/csw60-2016">CSW60 (2016)</a>

                                                </li>

                                                <li>

                                                    <a title="CSW59 / Beijing+20 (2015)" href="https://www.unwomen.org/en/csw/previous-sessions/csw59-2015">CSW59 / Beijing+20 (2015)</a>

                                                </li>

                                                <li>

                                                    <a title="CSW58 (2014)" href="https://www.unwomen.org/en/csw/previous-sessions/csw58-2014">CSW58 (2014)</a>

                                                </li>

                                                <li>

                                                    <a title="CSW57 (2013)" href="https://www.unwomen.org/en/csw/previous-sessions/csw57-2013">CSW57 (2013)</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Member States" href="https://www.unwomen.org/en/csw/member-states">Member States</a>

                                </li>

                                <li>

                                    <a title="NGO participation" href="https://www.unwomen.org/en/csw/ngo-participation">NGO participation</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="NGO participation" href="https://www.unwomen.org/en/csw/ngo-participation">NGO participation</a>

                                            </li>

                                                <li>

                                                    <a title="Eligibility" href="https://www.unwomen.org/en/csw/ngo-participation/eligibility">Eligibility</a>

                                                </li>

                                                <li>

                                                    <a title="Registration" href="https://www.unwomen.org/en/csw/ngo-participation/registration">Registration</a>

                                                </li>

                                                <li>

                                                    <a title="Opportunities for NGOs to address the Commission" href="https://www.unwomen.org/en/csw/ngo-participation/written-and-oral-statements">Opportunities for NGOs to address the Commission</a>

                                                </li>

                                                <li>

                                                    <a title="Accessibility" href="https://www.unwomen.org/en/csw/ngo-participation/accessibility">Accessibility</a>

                                                </li>

                                                <li>

                                                    <a title="NGO advisories" href="https://www.unwomen.org/en/csw/ngo-participation/ngo-advisories">NGO advisories</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="Communications procedure" href="https://www.unwomen.org/en/csw/communications-procedure">Communications procedure</a>

                                </li>

                                <li>

                                    <a title="Outcomes" href="https://www.unwomen.org/en/csw/outcomes">Outcomes</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Trust funds" href="https://www.unwomen.org/en/trust-funds">Trust funds</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Trust funds" href="https://www.unwomen.org/en/trust-funds">Trust funds</a>

                            </li>

                                <li>

                                    <a title="Fund for Gender Equality" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality">Fund for Gender Equality</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Fund for Gender Equality" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality">Fund for Gender Equality</a>

                                            </li>

                                                <li>

                                                    <a title="Our model" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/our-model">Our model</a>

                                                </li>

                                                <li>

                                                    <a title="Grant making" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/grantmaking">Grant making</a>

                                                </li>

                                                <li>

                                                    <a title="Accompaniment and growth" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/accompaniment-and-growth">Accompaniment and growth</a>

                                                </li>

                                                <li>

                                                    <a title="Results and impact" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/results-and-impact">Results and impact</a>

                                                </li>

                                                <li>

                                                    <a title="Knowledge and learning" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/knowledge-and-learning">Knowledge and learning</a>

                                                </li>

                                                <li>

                                                    <a title="Social innovation" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/social-innovation">Social innovation</a>

                                                </li>

                                                <li>

                                                    <a title="Join us" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/join-us">Join us</a>

                                                </li>

                                                <li>

                                                    <a title="Materials" href="https://www.unwomen.org/en/trust-funds/fund-for-gender-equality/materials">Materials</a>

                                                </li>

                                        </ul>

                                </li>

                                <li>

                                    <a title="UN Trust Fund to End Violence against Women" href="https://www.unwomen.org/en/trust-funds/un-trust-fund-to-end-violence-against-women">UN Trust Fund to End Violence against Women</a>

                                </li>

                        </ul>

                </li>

                <li>

                    <a title="Get involved" href="https://www.unwomen.org/en/get-involved">Get involved</a>

                        <ul class="dl-submenu">

                            <li>

                                <a title="Get involved" href="https://www.unwomen.org/en/get-involved">Get involved</a>

                            </li>

                                <li>

                                    <a title="Generation Equality" href="https://www.unwomen.org/en/get-involved/beijing-plus-25">Generation Equality</a>

                                        <ul class="dl-submenu">

                                            <li>

                                                <a title="Generation Equality" href="https://www.unwomen.org/en/get-involved/beijing-plus-25">Generation Equality</a>

                                            </li>

                                                <li>

                                                    <a title="About Generation Equality" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/about">About Generation Equality</a>

                                                </li>

                                                <li>

                                                    <a title="Generation Equality Forum" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/generation-equality-forum">Generation Equality Forum</a>

                                                </li>

                                                <li>

                                                    <a title="Action packs" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/take-action">Action packs</a>

                                                </li>

                                                <li>

                                                    <a title="Toolkit" href="https://www.unwomen.org/en/get-involved/beijing-plus-25/toolkit">Toolkit</a>

                                                </li>

                                        </ul>

                                </li>

                        </ul>

                </li>

        </ul>

    </div>

</div>

<!-- End Menu For Small Device-->







<div id="js-nav-primary" class="main-nav content-wrapper" style="visibility: visible;">



    <ul class="menuH">

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/about-us" title="About">About</a>

                



                    <ul>

                            <li>

                                <a class="arrow ga-event" title="About UN Women" href="https://www.unwomen.org/en/about-us/about-un-women" data-category="ui-nav" data-action="ui-mainnav-secondary">About UN Women</a>

                                    <ul>

                                            <li>

                                                <a title="&#8216;One Woman&#8217; &#8211; The UN Women song" href="https://www.unwomen.org/en/about-us/about-un-women/un-women-song" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">‘One Woman’ – The UN Women song</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Directorate" href="https://www.unwomen.org/en/about-us/directorate" data-category="ui-nav" data-action="ui-mainnav-secondary">Directorate</a>

                                    <ul>

                                            <li>

                                                <a title="Executive Director" href="https://www.unwomen.org/en/about-us/directorate/executive-director" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Executive Director</a>

                                            </li>

                                            <li>

                                                <a title="Deputy Executive Director for Normative Support, UN System Coordination and Programme Results" href="https://www.unwomen.org/en/about-us/directorate/ded-normative-support-un-system-coordination-and-programme-results" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Deputy Executive Director for Normative Support, UN System Coordination and Programme Results</a>

                                            </li>

                                            <li>

                                                <a title="Deputy Executive Director for Resource Management, Sustainability and Partnerships" href="https://www.unwomen.org/en/about-us/directorate/ded-resource-management-sustainability-and-partnerships" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Deputy Executive Director for Resource Management, Sustainability and Partnerships</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Governance" href="https://www.unwomen.org/en/about-us/governance" data-category="ui-nav" data-action="ui-mainnav-secondary">Governance</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Guiding documents" href="https://www.unwomen.org/en/about-us/guiding-documents" data-category="ui-nav" data-action="ui-mainnav-secondary">Guiding documents</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Accountability" href="https://www.unwomen.org/en/about-us/accountability" data-category="ui-nav" data-action="ui-mainnav-secondary">Accountability</a>

                                    <ul>

                                            <li>

                                                <a title="Evaluation" href="https://www.unwomen.org/en/about-us/accountability/evaluation" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Evaluation</a>

                                            </li>

                                            <li>

                                                <a title="Audit" href="https://www.unwomen.org/en/about-us/accountability/audit" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Audit</a>

                                            </li>

                                            <li>

                                                <a title="Report wrongdoing" href="https://www.unwomen.org/en/about-us/accountability/investigations" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Report wrongdoing</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Employment" href="https://www.unwomen.org/en/about-us/employment" data-category="ui-nav" data-action="ui-mainnav-secondary">Employment</a>

                                    <ul>

                                            <li>

                                                <a title="Internship programme" href="https://www.unwomen.org/en/about-us/employment/internship-programme" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Internship programme</a>

                                            </li>

                                            <li>

                                                <a title="UN Women Alumni Association" href="https://www.unwomen.org/en/about-us/employment/un-women-alumni-association" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">UN Women Alumni Association</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Procurement" href="https://www.unwomen.org/en/about-us/procurement" data-category="ui-nav" data-action="ui-mainnav-secondary">Procurement</a>

                                    <ul>

                                            <li>

                                                <a title="Procurement principles" href="https://www.unwomen.org/en/about-us/procurement/procurement-principles" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Procurement principles</a>

                                            </li>

                                            <li>

                                                <a title="Gender-responsive procurement" href="https://www.unwomen.org/en/about-us/procurement/gender-responsive-procurement" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Gender-responsive procurement</a>

                                            </li>

                                            <li>

                                                <a title="Doing business with UN Women" href="https://www.unwomen.org/en/about-us/procurement/doing-business-with-un-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Doing business with UN Women</a>

                                            </li>

                                            <li>

                                                <a title="Becoming a UN Women vendor" href="https://www.unwomen.org/en/about-us/procurement/how-to-become-a-un-women-supplier" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Becoming a UN Women vendor</a>

                                            </li>

                                            <li>

                                                <a title="Contract templates and general conditions of contract" href="https://www.unwomen.org/en/about-us/procurement/contract-templates-and-general-conditions-of-contract" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Contract templates and general conditions of contract</a>

                                            </li>

                                            <li>

                                                <a title="Vendor protest procedure" href="https://www.unwomen.org/en/about-us/procurement/vendor-protest-procedure" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Vendor protest procedure</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Contact us" href="https://www.unwomen.org/en/about-us/contact-us" data-category="ui-nav" data-action="ui-mainnav-secondary">Contact us</a>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/what-we-do" title="What&#32;we&amp;nbsp;do">What we&nbsp;do</a>

                



                    <ul>

                            <li>

                                <a class="arrow ga-event" title="Leadership and political participation" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation" data-category="ui-nav" data-action="ui-mainnav-secondary">Leadership and political participation</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Women&#8217;s movements" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/womens-movements" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women’s movements</a>

                                            </li>

                                            <li>

                                                <a title="Parliaments and local governance" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/parliaments-and-local-governance" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Parliaments and local governance</a>

                                            </li>

                                            <li>

                                                <a title="Constitutions and legal reform" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/constitutions-and-legal-reform" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Constitutions and legal reform</a>

                                            </li>

                                            <li>

                                                <a title="Elections" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/elections" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Elections</a>

                                            </li>

                                            <li>

                                                <a title="Media" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation/media" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Media</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Economic empowerment" href="https://www.unwomen.org/en/what-we-do/economic-empowerment" data-category="ui-nav" data-action="ui-mainnav-secondary">Economic empowerment</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and Figures" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and Figures</a>

                                            </li>

                                            <li>

                                                <a title="Global Norms and Standards" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global Norms and Standards</a>

                                            </li>

                                            <li>

                                                <a title="Macroeconomic policies and social protection" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/macroeconomics-policies-and-social-protection" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Macroeconomic policies and social protection</a>

                                            </li>

                                            <li>

                                                <a title="Sustainable Development and Climate Change" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/sustainable-development-and-climate-change" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Sustainable Development and Climate Change</a>

                                            </li>

                                            <li>

                                                <a title="Rural women" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/rural-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Rural women</a>

                                            </li>

                                            <li>

                                                <a title="Employment and migration" href="https://www.unwomen.org/en/what-we-do/economic-empowerment/employment-and-migration" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Employment and migration</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Ending violence against women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women" data-category="ui-nav" data-action="ui-mainnav-secondary">Ending violence against women</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="FAQs" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/faqs" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">FAQs</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Passing and implementing effective laws and policies" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/passing-strong-laws-and-policies" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Passing and implementing effective laws and policies</a>

                                            </li>

                                            <li>

                                                <a title="Services for all women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/services-for-all-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Services for all women</a>

                                            </li>

                                            <li>

                                                <a title="Increasing knowledge and awareness" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/increasing-knowledge-and-awareness" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Increasing knowledge and awareness</a>

                                            </li>

                                            <li>

                                                <a title="Creating safe public spaces" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/creating-safe-public-spaces" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Creating safe public spaces</a>

                                            </li>

                                            <li>

                                                <a title="Focusing on prevention to stop the violence" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/prevention" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Focusing on prevention to stop the violence</a>

                                            </li>

                                            <li>

                                                <a title="UNiTE campaign" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/take-action" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">UNiTE campaign</a>

                                            </li>

                                            <li>

                                                <a title="Spokesperson on Addressing Sexual Harassment" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/spokesperson-on-addressing-sexual-harassment" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Spokesperson on Addressing Sexual Harassment</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security" data-category="ui-nav" data-action="ui-mainnav-secondary">Peace and security</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/peace-and-security/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/peace-and-security/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Conflict prevention and resolution" href="https://www.unwomen.org/en/what-we-do/peace-and-security/conflict-prevention-and-resolution" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Conflict prevention and resolution</a>

                                            </li>

                                            <li>

                                                <a title="Building and sustaining peace" href="https://www.unwomen.org/en/what-we-do/peace-and-security/building-and-sustaining-peace" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Building and sustaining peace</a>

                                            </li>

                                            <li>

                                                <a title="Young women in peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security/young-women-in-peace-and-security" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Young women in peace and security</a>

                                            </li>

                                            <li>

                                                <a title="Rule of law: Justice and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security/rule-of-law-and-justice" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Rule of law: Justice and security</a>

                                            </li>

                                            <li>

                                                <a title="Women, peace, and security in the work of the UN Security Council" href="https://www.unwomen.org/en/what-we-do/peace-and-security/un-security-council" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women, peace, and security in the work of the UN Security Council</a>

                                            </li>

                                            <li>

                                                <a title="Preventing violent extremism and countering terrorism" href="https://www.unwomen.org/en/what-we-do/peace-and-security/preventing-violent-extremism" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Preventing violent extremism and countering terrorism</a>

                                            </li>

                                            <li>

                                                <a title="Planning and monitoring" href="https://www.unwomen.org/en/what-we-do/peace-and-security/planning-and-monitoring" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Planning and monitoring</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Humanitarian action" href="https://www.unwomen.org/en/what-we-do/humanitarian-action" data-category="ui-nav" data-action="ui-mainnav-secondary">Humanitarian action</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Humanitarian coordination" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/humanitarian-coordination" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Humanitarian coordination</a>

                                            </li>

                                            <li>

                                                <a title="Crisis response and recovery " href="https://www.unwomen.org/en/what-we-do/humanitarian-action/emergency-response" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Crisis response and recovery </a>

                                            </li>

                                            <li>

                                                <a title="Disaster risk reduction" href="https://www.unwomen.org/en/what-we-do/humanitarian-action/disaster-risk-reduction" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Disaster risk reduction</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Youth " href="https://www.unwomen.org/en/what-we-do/youth" data-category="ui-nav" data-action="ui-mainnav-secondary">Youth </a>

                                    <ul>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/youth/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Strengthening young women&#39;s leadership" href="https://www.unwomen.org/en/what-we-do/youth/strengthening-young-women-leadership" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Strengthening young women's leadership</a>

                                            </li>

                                            <li>

                                                <a title="Economic empowerment and skills development for young women" href="https://www.unwomen.org/en/what-we-do/youth/economic-empowerment-and-skills-development-for-young-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Economic empowerment and skills development for young women</a>

                                            </li>

                                            <li>

                                                <a title="Action on ending violence against young women and girls" href="https://www.unwomen.org/en/what-we-do/youth/action-on-ending-violence-against-young-women-and-girls" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Action on ending violence against young women and girls</a>

                                            </li>

                                            <li>

                                                <a title="Engaging boys and young men in gender equality" href="https://www.unwomen.org/en/what-we-do/youth/engaging-boys-and-young-men-in-gender-equality" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Engaging boys and young men in gender equality</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Governance and national planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning" data-category="ui-nav" data-action="ui-mainnav-secondary">Governance and national planning</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Inclusive National Planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/inclusive-national-planning" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Inclusive National Planning</a>

                                            </li>

                                            <li>

                                                <a title="Public Sector Reform" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/engaging-in-public-sector-reform" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Public Sector Reform</a>

                                            </li>

                                            <li>

                                                <a title="Tracking Investments" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning/tracking-investments" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Tracking Investments</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Sustainable development agenda" href="https://www.unwomen.org/en/what-we-do/post-2015" data-category="ui-nav" data-action="ui-mainnav-secondary">Sustainable development agenda</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="HIV and AIDS" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids" data-category="ui-nav" data-action="ui-mainnav-secondary">HIV and AIDS</a>

                                    <ul>

                                            <li>

                                                <a title="Facts and figures" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/facts-and-figures" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Facts and figures</a>

                                            </li>

                                            <li>

                                                <a title="Global norms and standards" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/global-norms-and-standards" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global norms and standards</a>

                                            </li>

                                            <li>

                                                <a title="Leadership and Participation" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/leadership-and-participation" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Leadership and Participation</a>

                                            </li>

                                            <li>

                                                <a title="National Planning" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/national-planning" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">National Planning</a>

                                            </li>

                                            <li>

                                                <a title="Violence against Women" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/violence-against-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Violence against Women</a>

                                            </li>

                                            <li>

                                                <a title="Access to Justice" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids/access-to-justice" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Access to Justice</a>

                                            </li>

                                    </ul>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/where-we-are" title="Where&#32;we&#32;are">Where we are</a>

                



                    <ul>

                            <li>

                                <a class="arrow ga-event" title="Africa" href="https://www.unwomen.org/en/where-we-are/africa" data-category="ui-nav" data-action="ui-mainnav-secondary">Africa</a>

                                    <ul>

                                            <li>

                                                <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/africa/regional-and-country-offices" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and country offices</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Americas and the Caribbean" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean" data-category="ui-nav" data-action="ui-mainnav-secondary">Americas and the Caribbean</a>

                                    <ul>

                                            <li>

                                                <a title="Regional and Country Offices" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean/regional-and-country-offices" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and Country Offices</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Arab States/&amp;#8203;North&amp;nbsp;Africa" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa" data-category="ui-nav" data-action="ui-mainnav-secondary">Arab States/&#8203;North&nbsp;Africa</a>

                                    <ul>

                                            <li>

                                                <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa/regional-and-country-offices" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and country offices</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Asia and the&#160;Pacific" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific" data-category="ui-nav" data-action="ui-mainnav-secondary">Asia and the Pacific</a>

                                    <ul>

                                            <li>

                                                <a title="Regional and country offices" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific/regional-and-country-offices" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and country offices</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Europe and Central&#160;Asia" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia" data-category="ui-nav" data-action="ui-mainnav-secondary">Europe and Central Asia</a>

                                    <ul>

                                            <li>

                                                <a title="Regional and Country Offices" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia/regional-and-country-offices" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and Country Offices</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Liaison offices" href="https://www.unwomen.org/en/where-we-are/liaison-offices" data-category="ui-nav" data-action="ui-mainnav-secondary">Liaison offices</a>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/how-we-work" title="How&#32;we&#32;work">How we work</a>

                



                    <ul>

                            <li>

                                <a class="arrow ga-event" title="Flagship programme initiatives" href="https://www.unwomen.org/en/how-we-work/flagship-programmes" data-category="ui-nav" data-action="ui-mainnav-secondary">Flagship programme initiatives</a>

                                    <ul>

                                            <li>

                                                <a title="Flagship programme: Making Every Woman and Girl Count " href="https://www.unwomen.org/en/how-we-work/flagship-programmes/making-every-woman-and-girl-count" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Flagship programme: Making Every Woman and Girl Count </a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Programme implementation" href="https://www.unwomen.org/en/how-we-work/programme-implementation" data-category="ui-nav" data-action="ui-mainnav-secondary">Programme implementation</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Innovation and technology" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology" data-category="ui-nav" data-action="ui-mainnav-secondary">Innovation and technology</a>

                                    <ul>

                                            <li>

                                                <a title="UN Women Global Innovation Coalition for Change" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology/un-women-global-innovation-coalition-for-change" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">UN Women Global Innovation Coalition for Change</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Intergovernmental support" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support" data-category="ui-nav" data-action="ui-mainnav-secondary">Intergovernmental support</a>

                                    <ul>

                                            <li>

                                                <a title="Commission on the Status of Women" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/commission-on-the-status-of-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Commission on the Status of Women</a>

                                            </li>

                                            <li>

                                                <a title="Climate change and the environment" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/climate-change-and-the-environment" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Climate change and the environment</a>

                                            </li>

                                            <li>

                                                <a title="High-Level Political Forum on Sustainable Development" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/hlpf-on-sustainable-development" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">High-Level Political Forum on Sustainable Development</a>

                                            </li>

                                            <li>

                                                <a title="Major resolutions" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/major-resolutions" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Major resolutions</a>

                                            </li>

                                            <li>

                                                <a title="Other Intergovernmental Processes" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/other-intergovernmental-processes" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Other Intergovernmental Processes</a>

                                            </li>

                                            <li>

                                                <a title="World Conferences on Women" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support/world-conferences-on-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">World Conferences on Women</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="UN system coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination" data-category="ui-nav" data-action="ui-mainnav-secondary">UN system coordination</a>

                                    <ul>

                                            <li>

                                                <a title="Global Coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/global-coordination" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Global Coordination</a>

                                            </li>

                                            <li>

                                                <a title="Regional and country coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/regional-and-country-coordination" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Regional and country coordination</a>

                                            </li>

                                            <li>

                                                <a title="Gender Mainstreaming" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/gender-mainstreaming" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Gender Mainstreaming</a>

                                            </li>

                                            <li>

                                                <a title="Promoting UN accountability" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/promoting-un-accountability" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Promoting UN accountability</a>

                                            </li>

                                            <li>

                                                <a title="Coordination resources" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/coordination-resources" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Coordination resources</a>

                                            </li>

                                            <li>

                                                <a title="UN Coordination Library" href="https://www.unwomen.org/en/how-we-work/un-system-coordination/library" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">UN Coordination Library</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Gender parity in the United Nations" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations" data-category="ui-nav" data-action="ui-mainnav-secondary">Gender parity in the United Nations</a>

                                    <ul>

                                            <li>

                                                <a title="System-wide strategy" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/system-wide-strategy" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">System-wide strategy</a>

                                            </li>

                                            <li>

                                                <a title="Gender Focal Points and Focal Points for Women" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/focal-points-for-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Gender Focal Points and Focal Points for Women</a>

                                            </li>

                                            <li>

                                                <a title="Data and statistics" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/current-status-of-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Data and statistics</a>

                                            </li>

                                            <li>

                                                <a title="Laws and policies" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/laws-and-policies" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Laws and policies</a>

                                            </li>

                                            <li>

                                                <a title="Strategies and tools" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/strategies-and-tools" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Strategies and tools</a>

                                            </li>

                                            <li>

                                                <a title="Reports and monitoring" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/reports-and-monitoring" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Reports and monitoring</a>

                                            </li>

                                            <li>

                                                <a title="Other resources" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations/other-resources" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Other resources</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Training for gender equality and women&#39;s empowerment" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training" data-category="ui-nav" data-action="ui-mainnav-secondary">Training for gender equality and women's empowerment</a>

                                    <ul>

                                            <li>

                                                <a title="Training Centre services" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training/training-centre-services" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Training Centre services</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Research and data" href="https://www.unwomen.org/en/how-we-work/research-and-data" data-category="ui-nav" data-action="ui-mainnav-secondary">Research and data</a>

                                    <ul>

                                            <li>

                                                <a title="Publications" href="https://www.unwomen.org/en/how-we-work/research-and-data/publications" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Publications</a>

                                            </li>

                                    </ul>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/partnerships" title="Partnerships">Partnerships</a>

                



                    <ul>

                            <li>

                                <a class="arrow ga-event" title="Government partners" href="https://www.unwomen.org/en/partnerships/donor-countries" data-category="ui-nav" data-action="ui-mainnav-secondary">Government partners</a>

                                    <ul>

                                            <li>

                                                <a title="Top government partners" href="https://www.unwomen.org/en/partnerships/donor-countries/top-donors" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Top government partners</a>

                                            </li>

                                            <li>

                                                <a title="Core resources" href="https://www.unwomen.org/en/partnerships/donor-countries/core-resources" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Core resources</a>

                                            </li>

                                            <li>

                                                <a title="Non-core resources" href="https://www.unwomen.org/en/partnerships/donor-countries/non-core-resources" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Non-core resources</a>

                                            </li>

                                            <li>

                                                <a title="Contribution trends" href="https://www.unwomen.org/en/partnerships/donor-countries/contribution-trends" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Contribution trends</a>

                                            </li>

                                            <li>

                                                <a title="Frequently asked questions" href="https://www.unwomen.org/en/partnerships/donor-countries/frequently-asked-questions" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Frequently asked questions</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="National mechanisms" href="https://www.unwomen.org/en/partnerships/national-mechanisms" data-category="ui-nav" data-action="ui-mainnav-secondary">National mechanisms</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Civil society" href="https://www.unwomen.org/en/partnerships/civil-society" data-category="ui-nav" data-action="ui-mainnav-secondary">Civil society</a>

                                    <ul>

                                            <li>

                                                <a title="Civil Society Advisory Groups" href="https://www.unwomen.org/en/partnerships/civil-society/civil-society-advisory-groups" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Civil Society Advisory Groups</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Businesses and philanthropies" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations" data-category="ui-nav" data-action="ui-mainnav-secondary">Businesses and philanthropies</a>

                                    <ul>

                                            <li>

                                                <a title="Benefits of partnering with UN&amp;nbsp;Women" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations/why-un-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Benefits of partnering with UN&nbsp;Women</a>

                                            </li>

                                            <li>

                                                <a title="Business and philanthropic partners" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations/major-partners" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Business and philanthropic partners</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="National Committees" href="https://www.unwomen.org/en/partnerships/national-committees" data-category="ui-nav" data-action="ui-mainnav-secondary">National Committees</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Goodwill Ambassadors" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors" data-category="ui-nav" data-action="ui-mainnav-secondary">Goodwill Ambassadors</a>

                                    <ul>

                                            <li>

                                                <a title="Danai Gurira" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/danai-gurira" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Danai Gurira</a>

                                            </li>

                                            <li>

                                                <a title="Nicole Kidman" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/nicole-kidman" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Nicole Kidman</a>

                                            </li>

                                            <li>

                                                <a title="Anne Hathaway" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/anne-hathaway" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Anne Hathaway</a>

                                            </li>

                                            <li>

                                                <a title="Marta Vieira da Silva" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/marta-vieira-da-silva" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Marta Vieira da Silva</a>

                                            </li>

                                            <li>

                                                <a title="Emma Watson" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/emma-watson" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Emma Watson</a>

                                            </li>

                                            <li>

                                                <a title="Farhan Akhtar" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/farhan-akhtar" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Farhan Akhtar</a>

                                            </li>

                                            <li>

                                                <a title="Princess Bajrakitiyabha Mahidol" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/princess-bajrakitiyabha" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Princess Bajrakitiyabha Mahidol</a>

                                            </li>

                                            <li>

                                                <a title="Tong Dawei" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/tong-dawei" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Tong Dawei</a>

                                            </li>

                                            <li>

                                                <a title="Jaha Dukureh" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/jaha-dukureh" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Jaha Dukureh</a>

                                            </li>

                                            <li>

                                                <a title="Muniba Mazari" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/muniba-mazari" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Muniba Mazari</a>

                                            </li>

                                            <li>

                                                <a title="Sania Mirza" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/sania-mirza" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Sania Mirza</a>

                                            </li>

                                            <li>

                                                <a title="Camila Pitanga" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/camila-pitanga" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Camila Pitanga</a>

                                            </li>

                                            <li>

                                                <a title="Hai Qing" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors/hai-qing" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Hai Qing</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="Media collaboration" href="https://www.unwomen.org/en/partnerships/media-collaboration" data-category="ui-nav" data-action="ui-mainnav-secondary">Media collaboration</a>

                                    <ul>

                                            <li>

                                                <a title="UN Women Media Compact" href="https://www.unwomen.org/en/partnerships/media-collaboration/media-compact" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">UN Women Media Compact</a>

                                            </li>

                                    </ul>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/news" title="News&#32;and&#32;events">News and events</a>

                



                    <ul>

                            <li>

                                <a class="ga-event" title="News" href="https://www.unwomen.org/en/news/stories" data-category="ui-nav" data-action="ui-mainnav-secondary">News</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Editorial series" href="https://www.unwomen.org/en/news/editorial-series" data-category="ui-nav" data-action="ui-mainnav-secondary">Editorial series</a>

                            </li>

                            <li>

                                <a class="arrow ga-event" title="In Focus" href="https://www.unwomen.org/en/news/in-focus" data-category="ui-nav" data-action="ui-mainnav-secondary">In Focus</a>

                                    <ul>

                                            <li>

                                                <a title="Gender equality matters in COVID-19 response" href="https://www.unwomen.org/en/news/in-focus/in-focus-gender-equality-in-covid-19-response" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Gender equality matters in COVID-19 response</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality Action Pack, March 2020" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-march-2020" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Generation Equality Action Pack, March 2020</a>

                                            </li>

                                            <li>

                                                <a title="International Women&#8217;s Day" href="https://www.unwomen.org/en/news/in-focus/international-womens-day" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Women’s Day</a>

                                            </li>

                                            <li>

                                                <a title="International Day of Women and Girls in Science" href="https://www.unwomen.org/en/news/in-focus/international-day-of-women-and-girls-in-science" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Day of Women and Girls in Science</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality action pack, January 2020" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-january-2020" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Generation Equality action pack, January 2020</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality action pack, December 2019" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-december-2019" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Generation Equality action pack, December 2019</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality action pack, November 2019: Generation Equality Stands against Rape" href="https://www.unwomen.org/en/news/in-focus/end-violence-against-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Generation Equality action pack, November 2019: Generation Equality Stands against Rape</a>

                                            </li>

                                            <li>

                                                <a title="Women, peace and security" href="https://www.unwomen.org/en/news/in-focus/women-peace-security" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women, peace and security</a>

                                            </li>

                                            <li>

                                                <a title="Generation Equality action pack, October 2019" href="https://www.unwomen.org/en/news/in-focus/generation-equality-action-pack-october-2019" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Generation Equality action pack, October 2019</a>

                                            </li>

                                            <li>

                                                <a title="International Day of Rural Women" href="https://www.unwomen.org/en/news/in-focus/rural-women-day" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Day of Rural Women</a>

                                            </li>

                                            <li>

                                                <a title="International Day of the Girl Child" href="https://www.unwomen.org/en/news/in-focus/girl-child" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Day of the Girl Child</a>

                                            </li>

                                            <li>

                                                <a title="74th session of the UN General Assembly" href="https://www.unwomen.org/en/news/in-focus/general-assembly" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">74th session of the UN General Assembly</a>

                                            </li>

                                            <li>

                                                <a title="World  Humanitarian Day" href="https://www.unwomen.org/en/news/in-focus/humanitarian-action" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">World  Humanitarian Day</a>

                                            </li>

                                            <li>

                                                <a title=" International Youth Day" href="https://www.unwomen.org/en/news/in-focus/youth" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary"> International Youth Day</a>

                                            </li>

                                            <li>

                                                <a title="International Day of the World&#8217;s Indigenous Peoples " href="https://www.unwomen.org/en/news/in-focus/indigenous-women" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Day of the World’s Indigenous Peoples </a>

                                            </li>

                                            <li>

                                                <a title="World Refugee Day" href="https://www.unwomen.org/en/news/in-focus/world-refugee-day" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">World Refugee Day</a>

                                            </li>

                                            <li>

                                                <a title="Women and girls in sport" href="https://www.unwomen.org/en/news/in-focus/women-and-sport" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women and girls in sport</a>

                                            </li>

                                            <li>

                                                <a title="International Girls in ICT Day" href="https://www.unwomen.org/en/news/in-focus/international-girls-in-ict-day" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">International Girls in ICT Day</a>

                                            </li>

                                            <li>

                                                <a title="CSW63" href="https://www.unwomen.org/en/news/in-focus/csw" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">CSW63</a>

                                            </li>

                                            <li>

                                                <a title="CSW62" href="https://www.unwomen.org/en/news/in-focus/csw62" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">CSW62</a>

                                            </li>

                                            <li>

                                                <a title="Women and the SDGs" href="https://www.unwomen.org/en/news/in-focus/women-and-the-sdgs" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women and the SDGs</a>

                                            </li>

                                            <li>

                                                <a title="In Focus: Climate action by, and for, women" href="https://www.unwomen.org/en/news/in-focus/climate-change" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">In Focus: Climate action by, and for, women</a>

                                            </li>

                                            <li>

                                                <a title="Indigenous women&#8217;s rights and activism" href="https://www.unwomen.org/en/news/in-focus/indigenous-womens-rights-and-activism" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Indigenous women’s rights and activism</a>

                                            </li>

                                            <li>

                                                <a title="Empowering women to conserve our oceans" href="https://www.unwomen.org/en/news/in-focus/empowering-women-to-conserve-our-oceans" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Empowering women to conserve our oceans</a>

                                            </li>

                                            <li>

                                                <a title="CSW61" href="https://www.unwomen.org/en/news/in-focus/csw61" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">CSW61</a>

                                            </li>

                                            <li>

                                                <a title="Women refugees and migrants" href="https://www.unwomen.org/en/news/in-focus/women-refugees-and-migrants" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Women refugees and migrants</a>

                                            </li>

                                            <li>

                                                <a title="CSW60" href="https://www.unwomen.org/en/news/in-focus/csw60" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">CSW60</a>

                                            </li>

                                            <li>

                                                <a title="Financing: Why it matters for women and girls" href="https://www.unwomen.org/en/news/in-focus/financing-for-gender-equality" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Financing: Why it matters for women and girls</a>

                                            </li>

                                            <li>

                                                <a title="Engaging Men" href="https://www.unwomen.org/en/news/in-focus/engaging-men" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Engaging Men</a>

                                            </li>

                                            <li>

                                                <a title="SIDS Conference" href="https://www.unwomen.org/en/news/in-focus/sids" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">SIDS Conference</a>

                                            </li>

                                            <li>

                                                <a title="MDG Momentum" href="https://www.unwomen.org/en/news/in-focus/mdg-momentum" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">MDG Momentum</a>

                                            </li>

                                            <li>

                                                <a title="Strengthening Women&#8217;s Access to Justice" href="https://www.unwomen.org/en/news/in-focus/strengthening-womens-access-to-justice" class="ga-event" data-category="ui-nav" data-action="ui-mainnav-secondary">Strengthening Women’s Access to Justice</a>

                                            </li>

                                    </ul>

                            </li>

                            <li>

                                <a class="ga-event" title="Events" href="https://www.unwomen.org/en/news/events" data-category="ui-nav" data-action="ui-mainnav-secondary">Events</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Media contacts" href="https://www.unwomen.org/en/news/media-contacts" data-category="ui-nav" data-action="ui-mainnav-secondary">Media contacts</a>

                            </li>

                    </ul>

            </li>

            <li>

                <a class="ga-event" data-action="ui-mainnav-primary" data-category="ui-nav" href="https://www.unwomen.org/en/digital-library" title="Digital&#32;library">Digital library</a>

                



                    <ul>

                            <li>

                                <a class="ga-event" title="Publications" href="https://www.unwomen.org/en/digital-library/publications" data-category="ui-nav" data-action="ui-mainnav-secondary">Publications</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Multimedia" href="https://www.unwomen.org/en/digital-library/multimedia" data-category="ui-nav" data-action="ui-mainnav-secondary">Multimedia</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Annual report" href="https://www.unwomen.org/en/digital-library/annual-report" data-category="ui-nav" data-action="ui-mainnav-secondary">Annual report</a>

                            </li>

                            <li>

                                <a class="ga-event" title="SDG monitoring report" href="https://www.unwomen.org/en/digital-library/sdg-report" data-category="ui-nav" data-action="ui-mainnav-secondary">SDG monitoring report</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Progress of the world&#8217;s women" href="https://www.unwomen.org/en/digital-library/progress-of-the-worlds-women" data-category="ui-nav" data-action="ui-mainnav-secondary">Progress of the world’s women</a>

                            </li>

                            <li>

                                <a class="ga-event" title="World survey on the role of women in development" href="https://www.unwomen.org/en/digital-library/world-survey-on-the-role-of-women-in-development" data-category="ui-nav" data-action="ui-mainnav-secondary">World survey on the role of women in development</a>

                            </li>

                            <li>

                                <a class="ga-event" title="Reprint permissions" href="https://www.unwomen.org/en/digital-library/reprint-permissions" data-category="ui-nav" data-action="ui-mainnav-secondary">Reprint permissions</a>

                            </li>

                            <li>

                                <a class="ga-event" title="GenderTerm" href="https://www.unwomen.org/en/digital-library/genderterm" data-category="ui-nav" data-action="ui-mainnav-secondary">GenderTerm</a>

                            </li>

                    </ul>

            </li>

    </ul>

</div>







<div class="breadcrumbs content-wrapper clearfix" role="navigation">

    <ul>

            <li class="">

                    <a href="/" title="Home">Home</a>



            </li>

            <li class="">

                    <a href="https://www.unwomen.org/en/what-we-do" title="What we&amp;nbsp;do">What we&nbsp;do</a>



            </li>

            <li class="current">

                    <a href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women" title="Ending violence against women">Ending violence against women</a>



            </li>

    </ul>

</div>



<div class="grid-group content-wrapper">

    <div class="grid grid-first grid-med grid-primarycol-xlrg">

        

<div class="article user-content">

        <h1>

            Facts and figures: Ending violence against women

        </h1>

    <h2>Various forms of violence</h2>

<ul>

    <li>It is estimated that 35 per cent of women worldwide have experienced either physical and/or sexual intimate partner violence or sexual violence by a non-partner (not including sexual harassment) at some point in their lives. However, some national studies show that up to 70 per cent of women have experienced physical and/or sexual violence from an intimate partner in their lifetime. Evidence shows that women who have experienced physical or sexual intimate partner violence report higher rates of depression, having an abortion and acquiring HIV, compared to women who have not [<a href="#notes">1</a>].</li>

    <li>Similar to data from other regions, in all four countries of a multi-country study from the Middle East and North Africa, men who witnessed their fathers using violence against their mothers, and men who experienced some form of violence at home as children, were significantly more likely to report perpetrating intimate partner violence in their adult relationships. For example, in Lebanon the likelihood of perpetrating physical violence was more than three times higher among men who had witnessed their fathers beating their mothers during childhood than those who did not [<a href="#notes">2</a>].</li>

    <li>It is estimated that of the 87,000 women who were intentionally killed in 2017 globally, more than half (50,000- 58 per cent) were killed by intimate partners or family members, meaning that 137 women across the world are killed by a member of their own family every day. More than a third (30,000) of the women intentionally killed in 2017 were killed by their current or former intimate partner [<a href="#notes">3</a>].

    </li>

</ul>

<div class="img-cap"><a href="https://interactive.unwomen.org/multimedia/infographic/violenceagainstwomen/en/index.html"><img alt="" width="675" src="https://www.unwomen.org/-/media/headquarters/images/sections/multimedia/2017/infographic-violence-against-women-opening-screen-en-960x700-loop.gif?la=en&vs=955" /></a>

<div class="caption"><a href="https://interactive.unwomen.org/multimedia/infographic/violenceagainstwomen/en/index.html">Learn more&raquo;</a></div>

</div>

<ul>

    <li>Adult women account for nearly half (49 per cent) of all human trafficking victims detected globally. Women and girls together account for 72 per cent, with girls representing more than three out of every four child trafficking victims. More than four out of every five trafficked women and nearly three out of every four trafficked girls are trafficked for the purpose of sexual exploitation [<a href="#notes">4</a>].</li>

    <li>It is estimated that there are 650 million women and girls in the world today who were married before age 18. During the past decade, the global rate of child marriage has declined. South Asia had the largest decline during this time, from 49 per cent to 30 per cent. Still, 12 million girls under 18 are married each year and in sub-Saharan Africa&mdash;where this harmful practice is most common&mdash;almost four out of 10 young women were married before their 18<sup>th</sup>&nbsp;birthday. Child marriage often results in early pregnancy and social isolation, interrupts schooling, limits the girl&rsquo;s opportunities and increases her risk of experiencing domestic violence [<a href="#notes">5</a>].</li>

    <li>At least 200 million women and girls aged 15-49 have undergone female genital mutilation in the 30 countries with representative data on prevalence. In most of these countries, the majority of girls were cut before age five. More than 20 million women and girls in just seven countries&nbsp;(Egypt, Sudan, Guinea, Djibouti, Kenya, Yemen and Nigeria)&nbsp;have undergone female genital mutilation by a health care provider.With population movement, female genital mutilation is becoming a practice with global dimensions, in particular among migrant and refugee women and girls [<a href="#notes">6</a>].</li>

    <li>Approximately 15 million adolescent girls (aged 15 to 19) worldwide have experienced forced sex (forced sexual intercourse or other sexual acts) at some point in their life. In the vast majority of countries, adolescent girls are most at risk of forced sex by a current/former husband, partner or boyfriend. Based on data from 30 countries, only one per cent ever sought professional help [<a href="#notes">7</a>].&nbsp;</li>

    <li>Globally, one out of three students (aged 11 and 13 to 15 years) have been bullied by their peers at school at least on one day in the past month, with girls and boys equally likely to experience bullying. However, boys are more likely to experience physical bullying than girls, and girls are more likely to experience psychological bullying, particularly being ignored or left out or subject to nasty rumours. Girls also report being made fun of because of how their face or body looks more frequently than boys. School-related gender-based violence is a major obstacle to universal schooling and the right to education for girls [<a href="h#notes">8</a>].</li>

    <li>Twenty-three per cent of female undergraduate university students reported having experienced sexual assault or sexual misconduct in a survey across 27 universities in the United States in 2015. Rates of reporting to campus officials, law enforcement or others ranged from five to 28 per cent, depending on the specific type of behavior [<a href="#notes">9</a>].</li>

    <li>One in 10 women in the European Union report having experienced cyber-harassment since the age of 15 (including having received unwanted, offensive sexually explicit emails or SMS messages, or offensive, inappropriate advances on social networking sites). The risk is highest among young women between 18 and 29 years of age [<a href="#notes">10</a>].</li>

    <li>In a multi-country study from the Middle East and North Africa, between 40 and 60 per cent of women said they had ever experienced street-based sexual harassment (mainly sexual comments, stalking/following, or staring/ogling), and 31 per cent to 64 per cent of men said they had ever carried out such acts. Younger men, men with more education, and men who experienced violence as children were more likely to engage in street sexual harassment [<a href="#notes">11</a>].</li>

    <li>Results from a national Australian survey show that almost two out of five women (39 per cent) aged 15 and older who have been in the workforce in the last five years have experienced sexual harassment in the workplace during that period, compared to one out of four (26 per cent) of their male counterparts. Regarding most common perpetrators, in almost four out of five cases (79 per cent) one or more of the perpetrators were male [<a href="#notes">12</a>].</li>

    <li>Eighty-two per cent of women parliamentarians who participated in a study conducted by the Inter-parliamentary Union in 39 countries across five regions reported having experienced some form of psychological violence (remarks, gestures and images of a sexist or humiliating sexual nature made against them or threats and/or mobbing) while serving their terms. They cited social media as the main channel through which such psychological violence is perpetrated; nearly half of those surveyed (44 per cent) reported having received death, rape, assault or abduction threats towards them or their families. Sixty-five per cent had been subjected to sexist remarks, primarily by male colleagues in parliament and from opposing parties as well as their own [<a href="#notes">13</a>].</li>

</ul>

<h2>Measures to address violence</h2>

<ul>

    <li>In the majority of countries with available data, less than 40 per cent of the women who experience violence seek help of any sort. Among women who do, most look to family and friends and very few look to formal institutions and mechanisms, such as police and health services. Less than 10 per cent of those women seeking help for experience of violence sought help by appealing to the police [<a href="#notes">14</a>].</li>

    <li>At least 144 countries have passed laws on domestic violence, and 154 have laws on sexual harassment. However, even when laws exist, this does not mean they are always compliant with international standards and recommendations or implemented [<a href="#notes">15</a>].</li>

    <li>Availability of data on violence against women has increased significantly in recent years. Comparable national prevalence data on intimate partner violence for the period 2005-2017 are available for 106 countries [<a href="#notes">16</a>].</li>

</ul>

<h2>Leaving no one behind</h2>

<ul>

    <li>The first large-scale research study of violence against women and girls in several areas of South Sudan that have known war and conflict for many years, showed that 33 per cent experienced sexual violence (including rape, attempted rape or any other unwanted sexual acts) by a non-partner (can include police officers or other armed actors, strangers or known persons). Seventy per cent or more of non-partner sexual assaults occurred during a direct experience of conflict (e.g. displacement, abduction or an attack on a survivor&rsquo;s village). Women who had directly experienced a conflict event (attack, injury, etc.) reported increased brutality and frequency [<a href="#notes">17</a>].</li>

    <li>In 2014, 23 per cent of non-heterosexual women (those who identified their sexual orientation as lesbian, bisexual or other) interviewed in the European Union indicated having experienced physical and/or sexual violence by both male and female non-partner perpetrators, compared with five per cent of heterosexual women [<a href="#notes">18</a>].</li>

    <li>In a national university student survey in Australia, 72 per cent of trans and gender diverse students (self-identifying their gender as &lsquo;indeterminate or unspecified&rsquo;, &lsquo;transgender&rsquo;, or &lsquo;other&rsquo;) reported having been sexually harassed at least once during 2016, in contrast to 63 per cent of female students, and 35 per cent of male students [<a href="#notes">19</a>].</li>

    <li>Data from female participants of prevention interventions in six low- and middle-income countries in Asia and Africa show that women with disabilities are two to four times more likely to experience partner violence than those without disabilities. Furthermore, the risk of experiencing all forms of partner violence and non-partner sexual violence increases with the severity of impairment. Qualitative data shows that disability-related stigma and discrimination, compounds women&rsquo;s vulnerability to violence and hinders their ability to seek help [<a href="#notes">20</a>].</li>

    <li>In a survey of 3,706 primary schoolchildren from Uganda, 24 per cent of 11 to 14-year-old girls with disabilities reported sexual violence at school, compared to 12 per cent of non-disabled girls [<a href="#notes">21</a>].</li>

    <li>In the 2017 National Youth Risk Behaviour survey in the US, nearly more than 9 per cent of high school girls who dated or went out with someone during the 12 months preceding the survey reported being physically hurt on purpose by someone they were dating or going out with during that period compared to nearly 7 per cent of high school boys; and almost 11 per cent reported having been forced to do sexual things they did not want to by someone they were dating or going out with compared to almost 3 per cent of high school boys [<a href="#notes">22</a>].</li>

    <li>In Australia, the prevalence of workplace sexual harassment in the past five years was substantially higher among people who identified as Aboriginal and Torres Strait Islander (53 per cent) compared with those who did not (32 per cent). There were no significant gender differences with 50 per cent of Aboriginal and Torres Strait Islander men and 55 per cent of Aboriginal and Torres Strait Islander women having experienced it in the last five years [<a href="#notes">23</a>].</li>

</ul>

<div id="notes" style="font-size: 0.8em;">

<h2>Notes</h2>

<p>[1] World Health Organization, Department of Reproductive Health and Research, London School of Hygiene and Tropical Medicine, South African Medical Research Council (2013).&nbsp;<a href="http://www.who.int/reproductivehealth/publications/violence/9789241564625/en/">Global and regional estimates of violence against women: prevalence and health effects of intimate partner violence and non-partner sexual violence</a>, p.2. For individual country information, see UN Women&nbsp;<a href="http://evaw-global-database.unwomen.org/en/countries">Global Database on Violence against Women</a>.</p>

<p>[2]&nbsp;Promundo and UN Women (2017).&nbsp;<a href="https://imagesmena.org/wp-content/uploads/sites/5/2017/05/IMAGES-MENA-Multi-Country-Report-EN-16May2017-web.pdf"><em>Understanding Masculinities: Results from the International Men and Gender Equality Survey (IMAGES) &ndash; Middle East and North Africa.</em></a><em>,</em>&nbsp;p. 16. For Lebanon information, see&nbsp;<a href="https://imagesmena.org/wp-content/uploads/sites/5/2017/12/IMAGES_Leb_Report_Final_Web_Dec13.pdf"><em>Understanding Masculinities: Results from the International Men and Gender Equality Survey (IMAGES) in Lebanon</em></a><em>,&nbsp;</em>p. 77.</p>

<p>[3] United Nations Office on Drugs and Crime (2019).&nbsp;<a href="https://www.unodc.org/documents/data-and-analysis/gsh/Booklet_5.pdf"><em>Global Study on Homicide 2019</em></a>, p. 10.</p>

<p>[4] UNODC (2018).&nbsp;<a href="https://www.unodc.org/documents/data-and-analysis/glotip/2018/GLOTiP_2018_BOOK_web_small.pdf">Global Report on Trafficking in Persons 201</a>8<em>, p. 25, 28</em>.</p>

<p>[5] UNICEF (2019).&nbsp;<a href="https://www.unicef.org/stories/child-marriage-around-world">Child marriage around the world- Infographic</a> and UNICEF (2017).&nbsp;<a href="https://data.unicef.org/resources/every-child-counted-status-data-children-sdgs/">Is every child counted? Status of Data for Children in the SDGs</a>,&nbsp;<em>p. 54.</em></p>

<p>[6] UNICEF (2019). <a href="https://www.unicef.org/stories/what-you-need-know-about-female-genital-mutilation">What you need to know about female genital mutilation- How the harmful practice affects millions of girls worldwide</a>; UNICEF (2016).&nbsp;<a href="http://www.unicef.org/media/files/FGMC_2016_brochure_final_UNICEF_SPREAD.pdf">Female Genital Mutilation/Cutting: A global concern</a>; and United Nations (2018).&nbsp;<a href="https://undocs.org/A/73/266"><em>Intensifying Global Efforts for the Elimination of Female Genital Mutilation, Report of the Secretary-General</em></a>, p.18.</p>

<p>[7] UNICEF (2017).&nbsp;<a href="https://www.unicef.org/publications/files/Violence_in_the_lives_of_children_and_adolescents.pdf"><em>A Familiar Face: Violence in the lives of children and adolescents</em></a><em>,&nbsp;</em>p. 73, 82.</p>

<p>[8] UNESCO (2019). <a href="https://unesdoc.unesco.org/ark:/48223/pf0000366483">Behind the numbers: ending school violence and bullying</a>, p.25-26; UNESCO (2018).&nbsp;<em><a href="http://www.ungei.org/global_status_on_school_violence(1).pdf">School violence and bullying: Global status and trends, drivers and consequences</a></em>, p. 4, 9; Education for All Global Monitoring Report (EFA GMR), UNESCO, United Nations Girls&rsquo; Education Initiative (UNGEI) (2015).&nbsp;<a href="http://www.ungei.org/resources/index_5968.html">School-related gender-based violence is preventing the achievement of quality education for all, Policy Paper 17</a>; and UNGEI (2014).&nbsp;<a href="http://www.ungei.org/resources/index_5903.html">End School-related gender-based violence (SRGBVB) infographic.</a></p>

<p>[9] Cantor, D., Fisher, B., Chibnall, S., Townsend, R., Lee, H., Bruce, C., and Thomas, G. (2015).&nbsp;<a href="https://www.aau.edu/key-issues/aau-climate-survey-sexual-assault-and-sexual-misconduct-2015">Report on the AAU Campus Climate Survey on Sexual Assault and Sexual Misconduct</a><em>,&nbsp;</em>p.13, 35.</p>

<p>[10] European Union Agency for Fundamental Rights (2014).&nbsp;<em><a href="https://fra.europa.eu/en/publication/2014/violence-against-women-eu-wide-survey-main-results-report">Violence against women: an EU-wide survey</a></em>, p. 104.</p>

<p>[11] Promundo and UN Women (2017).&nbsp;<a href="https://imagesmena.org/wp-content/uploads/sites/5/2017/05/IMAGES-MENA-Multi-Country-Report-EN-16May2017-web.pdf"><em>Understanding Masculinities: Results from the International Men and Gender Equality Survey (IMAGES) &ndash; Middle East and North Africa</em></a><em>,&nbsp;</em>p. 16.</p>

<p>[12] Australian Human Rights Commission (2018).&nbsp;<a href="https://www.humanrights.gov.au/sites/default/files/document/publication/AHRC_WORKPLACE_SH_2018.pdf"><em>Everyone&rsquo;s business: Fourth National Survey on Sexual Harassment in Australian Workplaces</em></a><em>,&nbsp;</em>p.8, 27.</p>

<p>[13] Inter-Parliamentary Union (2016).&nbsp;<a href="http://www.ipu.org/pdf/publications/issuesbrief-e.pdf">Sexism, harassment and violence against women parliamentarians</a>, p. 3.</p>

<p>[14] United Nations Economic and Social Affairs (2015).&nbsp;<a href="https://unstats.un.org/unsd/gender/worldswomen.html">The World&rsquo;s Women 2015, Trends and Statistics</a>,p. 159.</p>

<p>[15]&nbsp;World Bank Group (2018).&nbsp;<a href="https://wbl.worldbank.org/"><em>Women, Business and the Law 2018</em></a>, database.</p>

<p>[16]&nbsp;Inter-Agency and Expert Group on SDG Indicators. <a href="https://unstats.un.org/sdgs/indicators/database/">SDGs Global Database</a> (accessed on October 18, 2019).</p>

<p>[17] Global Women&rsquo;s Institute et al. (2017). No safe place: A lifetime of violence for conflict-affected women and girls in South Sudan, <a href="https://globalwomensinstitute.gwu.edu/sites/g/files/zaxdzs1356/f/downloads/No%20Safe%20Place_Summary_Report.pdf">Summary report</a>, p.12 and <a href="https://globalwomensinstitute.gwu.edu/sites/g/files/zaxdzs1356/f/downloads/No%20Safe%20Place_Policy_Brief_.pdf">Policy brief</a>, p.2.</p>

<p>[18] European Union Agency for Fundamental Rights (2014).&nbsp;<a href="http://fra.europa.eu/en/publication/2014/violence-against-women-eu-wide-survey-main-results-report">Violence against women: an EU-wide survey,</a>&nbsp;Annex 3, p. 184-188.</p>

<p>[19] Australian Human Rights Commission (2017).&nbsp;<a href="https://www.humanrights.gov.au/sites/default/files/document/publication/AHRC_2017_ChangeTheCourse_UniversityReport.pdf"><em>Change the Course: National Report on Sexual Assault and Sexual Harassment at Australian Universities</em></a>, p. 36</p>

<p>[20] Dunkle K., Van Der Heijden I., Stern E., and Chirwa E. (2018).&nbsp;<a href="https://www.whatworks.co.za/documents/publications/195-disability-brief-whatworks-23072018-web/file"><em>Disability and Violence against Women and Girls: Emerging Evidence from the What Works to Prevent Violence against Women and Girls Global Programme</em></a><em>,&nbsp;</em>p. 1-3.</p>

<p>[21] Devries, K., Kyegome, N., Zuurmond, M., Parkes, J., Child, J., Walakira, E. and Naker, D. (2014).&nbsp;<a href="https://bmcpublichealth.biomedcentral.com/articles/10.1186/1471-2458-14-1017">Violence against primary school children with disabilities in Uganda: a cross-sectional study</a>,&nbsp;p. 6</p>

<p>[22] CDC (2018).&nbsp;<a href="https://www.cdc.gov/healthyyouth/data/yrbs/pdf/2017/ss6708.pdf"><em>Youth Risk Behavior Surveillance&mdash;United States, 2017</em></a>, p.22,23.</p>

<p>[23] Australian Human Rights Commission (2018).&nbsp;<a href="https://www.humanrights.gov.au/sites/default/files/document/publication/AHRC_WORKPLACE_SH_2018.pdf"><em>Everyone&rsquo;s business: Fourth National Survey on Sexual Harassment in Australian Workplaces</em></a>, p.28.</p>

<p style="text-align: right;">[Page last updated in November 2019.]</p>

</div>

    

</div>



    </div>

    <!-- Ends grid grid-sm grid-primarycol-xlrg -->



    <div class="grid grid-med grid-modules">

        







<div class="social-shareThis" style="display: block;">

            <span class='st_facebook' st_title="Facts and figures: Ending violence against women" st_image="" st_summary=" Various forms of violence &#10; &#10; It is estimated that 35 per cent of women worldwide have experienced either physical and/or sexual intimate partner" st_url="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures"></span>

            <span class='st_twitter' st_title="Facts and figures: Ending violence against women" st_image="" st_summary=" Various forms of violence &#10; &#10; It is estimated that 35 per cent of women worldwide have experienced either physical and/or sexual intimate partner" st_url="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures" st_via="UN_Women"></span>

            <span class='st_email' dst_title="Facts and figures: Ending violence against women" st_image="" st_summary=" Various forms of violence &#10; &#10; It is estimated that 35 per cent of women worldwide have experienced either physical and/or sexual intimate partner" st_url="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures"></span>

            <span class='st_sharethis_custom share' st_title="Facts and figures: Ending violence against women" st_summary=" Various forms of violence &#10; &#10; It is estimated that 35 per cent of women worldwide have experienced either physical and/or sexual intimate partner" st_url="https://www.unwomen.org/en/what-we-do/ending-violence-against-women/facts-and-figures" st_image="">Share</span>

            <span>

            <a href="javascript:void(0);" class="print" onclick=" window.print(); " title="Print">Print</a>

        </span>

</div>





<div class="module">

    <div class="module-head-container-no-border media">

            <div class="img">

                <img src="https://www.unwomen.org/-/media/communications/icons/icon-module-news.jpg?vs=155&amp;h=32&amp;w=25&amp;la=en&amp;hash=ED51E952B258011109770AE5E0BA11A89DC8B5DA" alt="News" />

            </div>

        <h4 class="module-heading-lrg">

            Latest news

        </h4>

    </div>

    <div class="module-body-container">

        <ul class="media-list">

                <li class="">



                    <a href="https://www.unwomen.org/en/news/stories/2020/7/feature-women-in-iraq-play-vital-roles-in-the-countrys-covid-19-response" class="ga-event" data-category="ui-latestnews" data-action="ui-latestnews-1">Despite compounding challenges, women in Iraq play vital roles in the country’s COVID-19 response</a>

                </li>

                <li class="">



                    <a href="https://www.unwomen.org/en/news/stories/2020/7/op-ed-ed-phumzile-digitalization-in-the-decade-of-action" class="ga-event" data-category="ui-latestnews" data-action="ui-latestnews-2">Op-ed: The role of digitalization in the Decade of Action</a>

                </li>

                <li class="">



                    <a href="https://www.unwomen.org/en/news/stories/2020/6/news-tech-giants-provide-life-saving-information-during-covid-19" class="ga-event" data-category="ui-latestnews" data-action="ui-latestnews-3">Tech giants partner with UN Women to provide life-saving information to survivors of domestic violence during COVID-19</a>

                </li>

                <li class="">



                    <a href="https://www.unwomen.org/en/news/stories/2020/6/statement-inter-agency-statement-on-violence-against-women-and-girls--in-the-context-of-covid-19" class="ga-event" data-category="ui-latestnews" data-action="ui-latestnews-4">Inter-Agency statement on violence against women and girls in the context of COVID-19</a>

                </li>

        </ul>

    </div>

    <div class="module-footer">

<a href="https://www.unwomen.org/en/news" class="go-to" >More news &raquo;</a>    </div>

</div>





<div class="module-alt">

        <div class="module-head-container">

            <h5 class="module-heading-med">Orange the World: Generation Equality Stands against Rape</h5>

        </div>

        <div class="module-body-container">

            <a href="https://www.unwomen.org/en/news/in-focus/end-violence-against-women"><img height="137" alt="Generation Equality stands against rape" width="244" src="https://www.unwomen.org/-/media/headquarters/images/sections/news/in%20focus/evaw/banner_en_1200x675px.png?la=en&vs=1039" /></a>

<p>For the 16 Days of Activism against Gender-Based Violence, from 25 November to 10 December, and under the umbrella of the Generation Equality campaign to mark the 25th anniversary of the Beijing Declaration and Platform for Action, UN Secretary-General&rsquo;s UNiTE by 2030 to End Violence against Women campaign is calling upon people from all walks of life, across generations, to take our boldest stand yet against rape.

<span class="media"><a href="https://www.unwomen.org/en/news/in-focus/end-violence-against-women" class="more">More</a></span></p>

        </div>

</div>



<div style="margin-bottom: 35px;"><a href="https://untfevaw.rallyup.com/4d1896" title="Donate to end violence against women"><img height="300" alt="Donate to end violence against women" width="244" src="https://www.unwomen.org/-/media/headquarters/images/sidebarbanners/donate%20button%20244-300-en.jpg?la=en&amp;vs=3131" /></a></div>



        <div class="right-poster" style="margin-bottom: 35px;"></div>

    </div>

    <!-- Ends grid-modules -->

</div>



</div>

    

    <footer id="footer">

        





<div class="footer hq_footer content-wrapper">

    <h3 class="footer-heading">Follow us</h3>



    <ul class="interact">

            <li class="transition facebook" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-facebook.png?vs=3220')">



                <a href="http://www.facebook.com/unwomen" target="_top" class="inner" title="Facebook">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition instagram" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-instagram.png?vs=3653')">



                <a href="https://www.instagram.com/unwomen" target="_top" class="inner" title="Instagram">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition twitter" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-twitter.png?vs=4138')">



                <a href="http://www.twitter.com/UN_Women" target="_top" class="inner" title="Twitter">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition medium" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-medium.png?vs=3827')">



                <a href="https://medium.com/@UN_Women" target="_top" class="inner" title="Medium">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition linkedin" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-linkedin.png?vs=3709')">



                <a href="https://linkedin.com/company/un-women" target="_top" class="inner" title="LinkedIn">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition youtube" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-youtube.png?vs=3015')">



                <a href="https://www.youtube.com/UNWomen" target="_top" class="inner" title="YouTube">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition flickr" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-flickr.png?vs=3519')">



                <a href="https://www.flickr.com/photos/unwomen" target="_top" class="inner" title="Flickr">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition snapchat" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-snapchat.png?vs=4044')">



                <a href="https://www.snapchat.com/add/unwomen" target="_top" class="inner" title="Snapchat">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition rss" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-rss.png?vs=3845')">



                <a href="https://www.unwomen.org/en/rss-feeds" target="_top" class="inner" title="News feed">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

            <li class="transition donate" style="background-image: url('https://www.unwomen.org/-/media/communications/social%20media/footer-donate.png?vs=732')">



                <a href="https://donate.unwomen.org/en" target="_top" class="inner" title="Donate">

                    <div class="social-stats">

                        <span></span>

                    </div>

                </a>

            </li>

    </ul>

        <div class="footer-site-map">



            <ul class="site-map-root">

                    <li>

                        <h2 class="site-map-heading">



                            <a title="About" href="https://www.unwomen.org/en/about-us" class="ga-event" data-category="ui-nav" data-action="ui-footernav">About</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="About UN Women" href="https://www.unwomen.org/en/about-us/about-un-women" class="ga-event" data-category="ui-nav" data-action="ui-footernav">About UN Women</a>



                                </li>

                                <li>

                                    <a title="Directorate" href="https://www.unwomen.org/en/about-us/directorate" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Directorate</a>



                                </li>

                                <li>

                                    <a title="Governance" href="https://www.unwomen.org/en/about-us/governance" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Governance</a>



                                </li>

                                <li>

                                    <a title="Guiding documents" href="https://www.unwomen.org/en/about-us/guiding-documents" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Guiding documents</a>



                                </li>

                                <li>

                                    <a title="Accountability" href="https://www.unwomen.org/en/about-us/accountability" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Accountability</a>



                                </li>

                                <li>

                                    <a title="Employment" href="https://www.unwomen.org/en/about-us/employment" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Employment</a>



                                </li>

                                <li>

                                    <a title="Procurement" href="https://www.unwomen.org/en/about-us/procurement" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Procurement</a>



                                </li>

                                <li>

                                    <a title="Contact us" href="https://www.unwomen.org/en/about-us/contact-us" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Contact us</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="What we&amp;nbsp;do" href="https://www.unwomen.org/en/what-we-do" class="ga-event" data-category="ui-nav" data-action="ui-footernav">What we&nbsp;do</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="Leadership and political participation" href="https://www.unwomen.org/en/what-we-do/leadership-and-political-participation" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Leadership and political participation</a>



                                </li>

                                <li>

                                    <a title="Economic empowerment" href="https://www.unwomen.org/en/what-we-do/economic-empowerment" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Economic empowerment</a>



                                </li>

                                <li>

                                    <a title="Ending violence against women" href="https://www.unwomen.org/en/what-we-do/ending-violence-against-women" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Ending violence against women</a>



                                </li>

                                <li>

                                    <a title="Peace and security" href="https://www.unwomen.org/en/what-we-do/peace-and-security" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Peace and security</a>



                                </li>

                                <li>

                                    <a title="Humanitarian action" href="https://www.unwomen.org/en/what-we-do/humanitarian-action" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Humanitarian action</a>



                                </li>

                                <li>

                                    <a title="Youth " href="https://www.unwomen.org/en/what-we-do/youth" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Youth </a>



                                </li>

                                <li>

                                    <a title="Governance and national planning" href="https://www.unwomen.org/en/what-we-do/governance-and-national-planning" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Governance and national planning</a>



                                </li>

                                <li>

                                    <a title="Sustainable development agenda" href="https://www.unwomen.org/en/what-we-do/post-2015" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Sustainable development agenda</a>



                                </li>

                                <li>

                                    <a title="HIV and AIDS" href="https://www.unwomen.org/en/what-we-do/hiv-and-aids" class="ga-event" data-category="ui-nav" data-action="ui-footernav">HIV and AIDS</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="Where we are" href="https://www.unwomen.org/en/where-we-are" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Where we are</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="Africa" href="https://www.unwomen.org/en/where-we-are/africa" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Africa</a>



                                </li>

                                <li>

                                    <a title="Americas and the Caribbean" href="https://www.unwomen.org/en/where-we-are/americas-and-the-caribbean" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Americas and the Caribbean</a>



                                </li>

                                <li>

                                    <a title="Arab States/&amp;#8203;North&amp;nbsp;Africa" href="https://www.unwomen.org/en/where-we-are/arab-states-north-africa" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Arab States/&#8203;North&nbsp;Africa</a>



                                </li>

                                <li>

                                    <a title="Asia and the&#160;Pacific" href="https://www.unwomen.org/en/where-we-are/asia-and-the-pacific" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Asia and the Pacific</a>



                                </li>

                                <li>

                                    <a title="Europe and Central&#160;Asia" href="https://www.unwomen.org/en/where-we-are/europe-and-central-asia" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Europe and Central Asia</a>



                                </li>

                                <li>

                                    <a title="Liaison offices" href="https://www.unwomen.org/en/where-we-are/liaison-offices" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Liaison offices</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="How we work" href="https://www.unwomen.org/en/how-we-work" class="ga-event" data-category="ui-nav" data-action="ui-footernav">How we work</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="Flagship programme initiatives" href="https://www.unwomen.org/en/how-we-work/flagship-programmes" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Flagship programme initiatives</a>



                                </li>

                                <li>

                                    <a title="Programme implementation" href="https://www.unwomen.org/en/how-we-work/programme-implementation" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Programme implementation</a>



                                </li>

                                <li>

                                    <a title="Innovation and technology" href="https://www.unwomen.org/en/how-we-work/innovation-and-technology" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Innovation and technology</a>



                                </li>

                                <li>

                                    <a title="Intergovernmental support" href="https://www.unwomen.org/en/how-we-work/intergovernmental-support" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Intergovernmental support</a>



                                </li>

                                <li>

                                    <a title="UN system coordination" href="https://www.unwomen.org/en/how-we-work/un-system-coordination" class="ga-event" data-category="ui-nav" data-action="ui-footernav">UN system coordination</a>



                                </li>

                                <li>

                                    <a title="Gender parity in the United Nations" href="https://www.unwomen.org/en/how-we-work/gender-parity-in-the-united-nations" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Gender parity in the United Nations</a>



                                </li>

                                <li>

                                    <a title="Training for gender equality and women&#39;s empowerment" href="https://www.unwomen.org/en/how-we-work/capacity-development-and-training" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Training for gender equality and women's empowerment</a>



                                </li>

                                <li>

                                    <a title="Research and data" href="https://www.unwomen.org/en/how-we-work/research-and-data" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Research and data</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="Partnerships" href="https://www.unwomen.org/en/partnerships" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Partnerships</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="Government partners" href="https://www.unwomen.org/en/partnerships/donor-countries" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Government partners</a>



                                </li>

                                <li>

                                    <a title="National mechanisms" href="https://www.unwomen.org/en/partnerships/national-mechanisms" class="ga-event" data-category="ui-nav" data-action="ui-footernav">National mechanisms</a>



                                </li>

                                <li>

                                    <a title="Civil society" href="https://www.unwomen.org/en/partnerships/civil-society" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Civil society</a>



                                </li>

                                <li>

                                    <a title="Businesses and philanthropies" href="https://www.unwomen.org/en/partnerships/businesses-and-foundations" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Businesses and philanthropies</a>



                                </li>

                                <li>

                                    <a title="National Committees" href="https://www.unwomen.org/en/partnerships/national-committees" class="ga-event" data-category="ui-nav" data-action="ui-footernav">National Committees</a>



                                </li>

                                <li>

                                    <a title="Goodwill Ambassadors" href="https://www.unwomen.org/en/partnerships/goodwill-ambassadors" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Goodwill Ambassadors</a>



                                </li>

                                <li>

                                    <a title="Media collaboration" href="https://www.unwomen.org/en/partnerships/media-collaboration" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Media collaboration</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="News and events" href="https://www.unwomen.org/en/news" class="ga-event" data-category="ui-nav" data-action="ui-footernav">News and events</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="News" href="https://www.unwomen.org/en/news/stories" class="ga-event" data-category="ui-nav" data-action="ui-footernav">News</a>



                                </li>

                                <li>

                                    <a title="Editorial series" href="https://www.unwomen.org/en/news/editorial-series" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Editorial series</a>



                                </li>

                                <li>

                                    <a title="In Focus" href="https://www.unwomen.org/en/news/in-focus" class="ga-event" data-category="ui-nav" data-action="ui-footernav">In Focus</a>



                                </li>

                                <li>

                                    <a title="Events" href="https://www.unwomen.org/en/news/events" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Events</a>



                                </li>

                                <li>

                                    <a title="Media contacts" href="https://www.unwomen.org/en/news/media-contacts" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Media contacts</a>



                                </li>

                        </ul>

                    </li>

                    <li>

                        <h2 class="site-map-heading">



                            <a title="Digital library" href="https://www.unwomen.org/en/digital-library" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Digital library</a>

                        </h2>

                        <ul class="site-map-sub">

                                <li>

                                    <a title="Publications" href="https://www.unwomen.org/en/digital-library/publications" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Publications</a>



                                </li>

                                <li>

                                    <a title="Multimedia" href="https://www.unwomen.org/en/digital-library/multimedia" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Multimedia</a>



                                </li>

                                <li>

                                    <a title="Annual report" href="https://www.unwomen.org/en/digital-library/annual-report" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Annual report</a>



                                </li>

                                <li>

                                    <a title="SDG monitoring report" href="https://www.unwomen.org/en/digital-library/sdg-report" class="ga-event" data-category="ui-nav" data-action="ui-footernav">SDG monitoring report</a>



                                </li>

                                <li>

                                    <a title="Progress of the world&#8217;s women" href="https://www.unwomen.org/en/digital-library/progress-of-the-worlds-women" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Progress of the world’s women</a>



                                </li>

                                <li>

                                    <a title="World survey on the role of women in development" href="https://www.unwomen.org/en/digital-library/world-survey-on-the-role-of-women-in-development" class="ga-event" data-category="ui-nav" data-action="ui-footernav">World survey on the role of women in development</a>



                                </li>

                                <li>

                                    <a title="Reprint permissions" href="https://www.unwomen.org/en/digital-library/reprint-permissions" class="ga-event" data-category="ui-nav" data-action="ui-footernav">Reprint permissions</a>



                                </li>

                                <li>

                                    <a title="GenderTerm" href="https://www.unwomen.org/en/digital-library/genderterm" class="ga-event" data-category="ui-nav" data-action="ui-footernav">GenderTerm</a>



                                </li>

                        </ul>

                    </li>

            </ul>

        </div>



    <div class="masthead-container">

        <h1 class="Footer-logo">

            <a href="https://www.unwomen.org/en">

                <img src="https://www.unwomen.org/-/media/un%20women%20logos/footer%20logo/un-women-logo-mini-en.png?vs=5802&amp;h=31&amp;w=109&amp;la=en&amp;hash=8BCDE869C0252DA758564C491BED5FF8388D1DB1" alt="UN&#32;Women" />

            </a>

            <a class="footer-copyright" href="/en/about-the-website/copyright">Copyright</a> &copy; <a href="https://www.unwomen.org/en">UN Women</a>

        </h1>

    </div>



    <ul class="nav tertiary">

                    <li class='first'><a href="https://www.unwomen.org/en/about-the-website/terms-of-use">Terms of use</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-the-website/privacy-notice">Privacy notice</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-us/accountability/investigations">Report wrongdoing</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-the-website/information-security">Information security</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-us/employment">Employment</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-us/procurement">Procurement</a></li>

                    <li class=''><a href="https://www.unwomen.org/en/about-us/contact-us">Contact us</a></li>



    </ul>



</div>



    </footer>



        <!-- Placed at the end of the document so the pages load faster -->





<script src="https://www.unwomen.org/Scripts/communications/json2.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/jquery.dlmenu.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/fancybox/jquery.fancybox.pack.js?d=2020-06-25T05:51:50" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/jquery.cookie.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/jquery.qtip.min.js?d=2020-06-25T05:51:42" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/jquery.easing.1.3.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/counter.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/inview.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/URI.js?d=2020-06-25T05:51:42" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/Utils.js?d=2020-06-25T05:51:42" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/jquery.unwomen-table-formatters.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/jquery.unwomen-caption-formatter.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/global.js?d=2020-06-25T05:51:50" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/breakingNews.js?d=2020-06-25T05:51:42" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/tooltipster.bundle.min.js?d=2020-06-25T05:51:46" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/MixMedia/jquery.bxslider.js?d=2020-06-25T05:52:00" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/MixMedia/jquery.bxslider_custom.js?d=2020-06-25T05:51:50" type="text/javascript"></script>

<script src="https://www.unwomen.org/Scripts/communications/purify.min.js?d=2020-06-25T05:51:44" type="text/javascript"></script>

<script type="text/javascript">var switchTo5x = true;</script>

    <script type="text/javascript" src="https://ws.sharethis.com/button/buttons.js"></script>

<script type="text/javascript">

 stLight.options(

    {

        publisher: "c33bc976-8a44-4113-aa0d-a770b17574bf",

        doNotHash: false,

        doNotCopy: false,

        //hashAddressBar: true,

        shorten: true,

        onhover: false

    }); </script></body>

</html>

"""
soup = BeautifulSoup(html, 'lxml')

soup.find_all('div')
listy = []

for i in soup.find_all('li'):

    listy.append(i.text)

listy
facts = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]', '[11]']

facts_list = []

for i in listy:

    for f in facts:

        if f in i:

            facts_list.append(i)

        else:

            pass

facts_list
for i in facts_list:

    for f in facts:

        if f in i:

            i.strip

        else:

            pass

facts_list
facts_df = pd.DataFrame({'Fact': facts_list})

facts_df
vectorizer = TfidfVectorizer(use_idf = True)

X = vectorizer.fit_transform(facts_df['Fact'])

print(vectorizer.get_feature_names())

print('-----------------------------')

print(X.shape)
first_doc = X[0]

df = pd.DataFrame(first_doc.T.todense(), index = vectorizer.get_feature_names(), 

                  columns = ['Tf-idf score'])

df.loc[df['Tf-idf score'] >0 ].sort_values(by = 'Tf-idf score', ascending = False)
second_doc = X[1]

df_1 = pd.DataFrame(second_doc.T.todense(), index = vectorizer.get_feature_names(), 

                  columns = ['Tf-idf score'])

df_1.loc[df_1['Tf-idf score'] > 0].sort_values(by = 'Tf-idf score', ascending = False)
third_doc = X[2]

df_2 = pd.DataFrame(third_doc.T.todense(), index = vectorizer.get_feature_names(), 

                  columns = ['Tf-idf score'])

df_2.loc[df_2['Tf-idf score'] > 0].sort_values(by = 'Tf-idf score', ascending = False)
def key_terms(data):

    for i, j in enumerate(data):

        df = pd.DataFrame(j.T.todense(), index = vectorizer.get_feature_names(), 

                  columns = ['Tf-idf score'])

        final_df = df.loc[df['Tf-idf score'] > 0].sort_values(by = 'Tf-idf score', ascending = False)[0:10]

        

        print('For the document {}, these are the most import terms: {}'.format(i,final_df.index))

        print('-'*110)
key_terms = key_terms(X)
africa_df = pd.DataFrame({'Fact':[facts_df['Fact'][1], facts_df['Fact'][4], facts_df['Fact'][5],

                          facts_df['Fact'][10]]})

africa_df
africa_df['Abuse'] = ['physical', 'child marriage', 'genital mutilation', 'sexual']

africa_df
africa_df.Fact[0]
africa_df.Fact[1]
africa_df.Fact[2]
africa_df.Fact[3]
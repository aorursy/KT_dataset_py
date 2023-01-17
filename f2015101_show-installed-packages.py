try:
    from pip._internal.utils.misc import get_installed_distributions
except ImportError:  # pip<10
    from pip import get_installed_distributions
for package in sorted(get_installed_distributions(), key=lambda package: package.project_name):
    print("{} ({})".format(package.project_name, package.version))